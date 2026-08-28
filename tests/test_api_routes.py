from deep_research.api.app import create_app
from tests.conftest import FakeGateway


def test_review_and_report_routes_are_exposed() -> None:
    paths = create_app(FakeGateway()).openapi()["paths"]

    assert "/v1/reviewer/review" in paths
    assert "/v1/report/generate" in paths
    assert "/v1/researcher/research" in paths
