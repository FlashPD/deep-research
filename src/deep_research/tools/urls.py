from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
}


def canonicalize_url(url: str) -> str:
    """Normalize a public URL so one page maps to one stable source identity."""
    parsed = urlsplit(url)
    filtered_query = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name.casefold() not in _TRACKING_PARAMETERS and not name.casefold().startswith("utm_")
    ]
    path = parsed.path or "/"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            path,
            urlencode(filtered_query),
            "",
        )
    )
