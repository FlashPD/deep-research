import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from deep_research.auth.cognito import DevelopmentAuthenticator
from deep_research.client.polling import RunEventPoller
from deep_research.contracts.clarification import (
    AnswerType,
    ClarificationAnswer,
    ClarificationQuestion,
)
from deep_research.contracts.jobs import JobPhase
from deep_research.contracts.runs import (
    CreateRunRequest,
    PlanApprovalRequest,
    Principal,
    ResearchRun,
    RunState,
)
from deep_research.dev_app import LocalDevelopmentRuntime, build_local_runtime
from deep_research.settings import AppSettings
from deep_research.worker import make_phase_job

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

NON_INTERACTIVE_MESSAGE = (
    "stdin is not interactive; pass --auto-approve-plan to run non-interactively "
    "(clarification questions still require a terminal)"
)
INPUT_CLOSED_MESSAGE = "Interactive input was required but stdin is closed; cancelling the run."


def _print_flushed(text: str) -> None:
    print(text, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research-run",
        description="Run the complete local deep-research workflow.",
    )
    parser.add_argument("topic", help="Research topic or question")
    parser.add_argument("--depth", choices=("quick", "standard", "deep"), default="standard")
    parser.add_argument("--provider", choices=("openai", "anthropic", "bedrock"))
    parser.add_argument("--output", type=Path, default=Path("research-reports"))
    parser.add_argument("--auto-approve-plan", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log model routing, provider fallbacks, and adapter failures to stderr.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def main(argv: Sequence[str] | None = None, *, stdin_is_tty: bool | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    interactive = sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty
    if not args.auto_approve_plan and not interactive:
        print(NON_INTERACTIVE_MESSAGE, file=sys.stderr)
        raise SystemExit(2)
    settings = AppSettings(
        **({"model_provider": args.provider} if args.provider is not None else {})
    )
    try:
        runtime = build_local_runtime(settings)
        exit_code = asyncio.run(run_cli(args, runtime))
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(exit_code)


async def run_cli(
    args: argparse.Namespace,
    runtime: LocalDevelopmentRuntime,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = _print_flushed,
) -> int:
    principal = await DevelopmentAuthenticator().authenticate(None)
    worker_task = asyncio.create_task(
        runtime.worker.run_forever(), name="deep-research-cli-worker"
    )
    run: ResearchRun | None = None
    try:
        run = await runtime.control.create_run(
            principal,
            CreateRunRequest(topic=args.topic, depth=args.depth),
            idempotency_key=_key("cli-create"),
        )
        run = await runtime.control.start_run(
            principal, run.run_id, idempotency_key=_key("cli-start")
        )
        await runtime.dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))
        output_fn(f"Run {run.run_id} started with {args.depth} depth.")
        return await _drive_run(
            args,
            runtime,
            principal,
            run.run_id,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        if run is not None:
            await asyncio.shield(_cooperative_cancel(runtime, principal, run.run_id, output_fn))
        return 130
    finally:
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task


async def _drive_run(
    args: argparse.Namespace,
    runtime: LocalDevelopmentRuntime,
    principal: Principal,
    run_id: str,
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> int:
    poller = RunEventPoller()
    while True:
        for event in await poller.poll(runtime.control, principal, run_id):
            payload = json.dumps(event.payload, sort_keys=True, separators=(",", ":"))
            output_fn(f"[{event.cursor}] {event.event_type} {payload}")

        run = await runtime.control.get_run(principal, run_id)
        checkpoint = run.graph_checkpoint
        if (
            run.state is RunState.CLARIFYING
            and checkpoint.pending_clarification_questions
            and not checkpoint.submitted_clarification_answers
        ):
            output_fn(f"\nClarification round {checkpoint.clarification_round}/3")
            try:
                answers = _collect_answers(
                    checkpoint.pending_clarification_questions,
                    input_fn=input_fn,
                    output_fn=output_fn,
                )
            except EOFError:
                output_fn(INPUT_CLOSED_MESSAGE)
                await _cooperative_cancel(runtime, principal, run_id, output_fn)
                return 2
            run = await runtime.control.submit_clarification_answers(
                principal,
                run_id,
                answers,
                round_number=checkpoint.clarification_round,
                idempotency_key=_key(f"cli-clarify-{checkpoint.clarification_round}"),
            )
            await runtime.dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))
            continue

        if run.state is RunState.AWAITING_PLAN_APPROVAL and run.plan is not None:
            output_fn("\nGenerated research plan:\n")
            output_fn(run.plan.model_dump_json(indent=2))
            try:
                approved = args.auto_approve_plan or _confirm(
                    "Approve this exact plan? [y/N] ", input_fn
                )
            except EOFError:
                output_fn(INPUT_CLOSED_MESSAGE)
                await _cooperative_cancel(runtime, principal, run_id, output_fn)
                return 2
            if not approved:
                await _cooperative_cancel(runtime, principal, run_id, output_fn)
                return 1
            run = await runtime.control.approve_plan(
                principal,
                run_id,
                PlanApprovalRequest(
                    version=run.plan.version,
                    content_hash=run.plan.content_hash,
                ),
                idempotency_key=_key("cli-approve-plan"),
            )
            await runtime.dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))
            output_fn(
                f"Approved plan version {run.approved_plan_version} "
                f"({run.approved_plan_hash})."
            )
            continue

        if run.state is RunState.COMPLETED:
            for event in await poller.poll(runtime.control, principal, run_id):
                output_fn(f"[{event.cursor}] {event.event_type}")
            return _write_completed_run(args.output, run, output_fn)
        if run.state in {RunState.FAILED, RunState.CANCELLED, RunState.EXPIRED}:
            output_fn(f"Run ended in state {run.state.value}.")
            if run.failure_code:
                output_fn(f"Failure: {run.failure_code}: {run.failure_message or 'No detail'}")
            _display_limitations(run, output_fn)
            return 1
        await asyncio.sleep(0.25)


def _collect_answers(
    questions: list[ClarificationQuestion], *, input_fn: InputFn, output_fn: OutputFn
) -> list[ClarificationAnswer]:
    answers: list[ClarificationAnswer] = []
    for question in questions:
        output_fn(f"\n{question.question}")
        output_fn(f"Why this matters: {question.rationale}")
        answer = _prompt_for_answer(question, input_fn=input_fn, output_fn=output_fn)
        if answer is not None:
            answers.append(ClarificationAnswer(question_id=question.id, value=answer))
    return answers


def _prompt_for_answer(
    question: ClarificationQuestion, *, input_fn: InputFn, output_fn: OutputFn
) -> str | list[str] | bool | None:
    if question.expected_answer_type is AnswerType.CONFIRMATION:
        while True:
            raw = input_fn("Answer [y/n]: ").strip().casefold()
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            if not raw and not question.required:
                return None
            output_fn("Please answer yes or no.")

    if question.expected_answer_type in {AnswerType.SINGLE_SELECT, AnswerType.MULTI_SELECT}:
        for index, option in enumerate(question.options, start=1):
            output_fn(f"  {index}. {option}")
        while True:
            label = (
                "Choose one: "
                if question.expected_answer_type is AnswerType.SINGLE_SELECT
                else "Choose one or more numbers separated by commas: "
            )
            raw = input_fn(label).strip()
            if not raw and not question.required:
                return None
            try:
                indexes = [int(item.strip()) for item in raw.split(",")]
                selected = [question.options[index - 1] for index in indexes]
                if any(index < 1 or index > len(question.options) for index in indexes):
                    raise ValueError
                if len(indexes) != len(set(indexes)):
                    raise ValueError
                if question.expected_answer_type is AnswerType.SINGLE_SELECT:
                    if len(selected) != 1:
                        raise ValueError
                    return selected[0]
                return selected
            except (ValueError, IndexError):
                output_fn("Please choose from the listed option numbers.")

    while True:
        raw = input_fn("Answer: ").strip()
        if raw:
            return raw
        if not question.required:
            return None
        output_fn("This question requires an answer.")


def _confirm(prompt: str, input_fn: InputFn) -> bool:
    return input_fn(prompt).strip().casefold() in {"y", "yes"}


async def _cooperative_cancel(
    runtime: LocalDevelopmentRuntime,
    principal: Principal,
    run_id: str,
    output_fn: OutputFn,
) -> None:
    run = await runtime.control.get_run(principal, run_id)
    if not run.state.terminal:
        await runtime.control.cancel(
            principal, run_id, idempotency_key=_key("cli-cancel")
        )
    output_fn(f"Cancellation requested for run {run_id}.")


def _write_completed_run(output: Path, run: ResearchRun, output_fn: OutputFn) -> int:
    report = run.graph_checkpoint.report
    if report is None:
        output_fn("Run completed without a report artifact.")
        return 1
    target = output.expanduser()
    if target.suffix.casefold() != ".md":
        target = target / f"{run.run_id}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.markdown, encoding="utf-8")
    output_fn(f"\nReport written to {target.resolve()}")
    _display_limitations(run, output_fn)
    questions = run.graph_checkpoint.questions
    if questions is not None:
        output_fn("\nFollow-up questions:")
        for item in questions.questions:
            output_fn(f"  {item.priority}. {item.question}")
    return 0


def _display_limitations(run: ResearchRun, output_fn: OutputFn) -> None:
    limitations = list(run.graph_checkpoint.limitations)
    review = run.graph_checkpoint.review
    if review is not None:
        limitations.extend(review.limitations)
    limitations = list(dict.fromkeys(limitations))
    if limitations:
        output_fn("\nReviewer and workflow limitations:")
        for limitation in limitations:
            output_fn(f"  - {limitation}")


def _key(prefix: str) -> str:
    return f"{prefix}:{uuid4().hex}"


if __name__ == "__main__":  # pragma: no cover
    main()
