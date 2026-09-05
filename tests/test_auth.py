import json
from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from deep_research.auth.cognito import AuthenticationError, CognitoAuthenticator


@pytest.fixture
def token_material():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": "test-key", "alg": "RS256", "use": "sig"})
    return private_key, jwk


def _token(private_key, **overrides) -> str:
    now = datetime.now(UTC)
    claims = {
        "sub": "user-123",
        "iss": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pool",
        "exp": now + timedelta(minutes=5),
        "iat": now,
        "client_id": "app-client-123",
        "token_use": "access",
        "scope": "openid research:run",
        "custom:tenant_id": "tenant-456",
        **overrides,
    }
    return jwt.encode(
        claims,
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )


def _authenticator(jwk: dict, requests: list[httpx.Request]) -> CognitoAuthenticator:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"keys": [jwk]})

    return CognitoAuthenticator(
        issuer="https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pool",
        client_id="app-client-123",
        required_scopes=frozenset({"research:run"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_cognito_access_token_is_verified_and_jwks_is_cached(token_material) -> None:
    private_key, jwk = token_material
    requests: list[httpx.Request] = []
    authenticator = _authenticator(jwk, requests)
    token = _token(private_key)

    first = await authenticator.authenticate(f"Bearer {token}")
    second = await authenticator.authenticate(f"Bearer {token}")

    assert first == second
    assert first.subject == "user-123"
    assert first.tenant_id == "tenant-456"
    assert "research:run" in first.scopes
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "status_code"),
    [
        ({"client_id": "another-client"}, 401),
        ({"token_use": "id"}, 401),
        ({"scope": "openid"}, 403),
        ({"iss": "https://attacker.example"}, 401),
        ({"exp": datetime.now(UTC) - timedelta(minutes=5)}, 401),
    ],
)
async def test_cognito_rejects_invalid_claims(
    token_material, overrides: dict, status_code: int
) -> None:
    private_key, jwk = token_material
    authenticator = _authenticator(jwk, [])

    with pytest.raises(AuthenticationError) as error:
        await authenticator.authenticate(f"Bearer {_token(private_key, **overrides)}")

    assert error.value.status_code == status_code


@pytest.mark.asyncio
async def test_cognito_requires_a_bearer_token(token_material) -> None:
    _, jwk = token_material
    authenticator = _authenticator(jwk, [])

    with pytest.raises(AuthenticationError, match="missing bearer token"):
        await authenticator.authenticate(None)


@pytest.mark.asyncio
async def test_invalid_jwks_document_becomes_an_authentication_error() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"not-json"))
    )
    authenticator = CognitoAuthenticator(
        issuer="https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pool",
        client_id="app-client-123",
        required_scopes=frozenset(),
        client=client,
    )
    token = "eyJhbGciOiJSUzI1NiIsImtpZCI6InRlc3Qta2V5In0.e30.signature"

    with pytest.raises(AuthenticationError):
        await authenticator.authenticate(f"Bearer {token}")
