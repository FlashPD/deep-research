import asyncio
import time
from typing import Any, Protocol

import httpx
import jwt

from deep_research.contracts.runs import Principal
from deep_research.settings import AppSettings


class AuthenticationError(ValueError):
    def __init__(self, message: str, *, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


class RequestAuthenticator(Protocol):
    async def authenticate(self, authorization: str | None) -> Principal: ...


class DevelopmentAuthenticator:
    """Explicit local mode with a fixed identity and no client-controlled owner fields."""

    def __init__(
        self,
        *,
        subject: str = "local-user",
        tenant_id: str = "local-tenant",
        scopes: frozenset[str] = frozenset({"research:run"}),
    ) -> None:
        self._principal = Principal(
            subject=subject,
            tenant_id=tenant_id,
            scopes=scopes,
        )

    async def authenticate(self, authorization: str | None) -> Principal:
        return self._principal


class CognitoAuthenticator:
    def __init__(
        self,
        *,
        issuer: str,
        client_id: str,
        required_scopes: frozenset[str],
        tenant_claim: str = "custom:tenant_id",
        jwks_cache_seconds: int = 3_600,
        clock_skew_seconds: int = 30,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not issuer.startswith("https://") or not client_id:
            raise ValueError("Cognito issuer and client_id are required")
        self._issuer = issuer.rstrip("/")
        self._client_id = client_id
        self._required_scopes = required_scopes
        self._tenant_claim = tenant_claim
        self._cache_seconds = jwks_cache_seconds
        self._clock_skew = clock_skew_seconds
        self._client = client
        self._keys: dict[str, Any] = {}
        self._keys_expire_at = 0.0
        self._lock = asyncio.Lock()

    async def authenticate(self, authorization: str | None) -> Principal:
        token = _bearer_token(authorization)
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthenticationError("invalid bearer token") from exc
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
            raise AuthenticationError("unsupported token signature header")
        key = await self._key_for(header["kid"])
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                issuer=self._issuer,
                options={
                    "verify_aud": False,
                    "require": [
                        "sub",
                        "iss",
                        "exp",
                        "iat",
                        "client_id",
                        "token_use",
                    ],
                },
                leeway=self._clock_skew,
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError("token validation failed") from exc
        if claims.get("token_use") != "access":
            raise AuthenticationError("token is not an access token")
        if claims.get("client_id") != self._client_id:
            raise AuthenticationError("token was issued to a different app client")
        scopes = frozenset(str(claims.get("scope", "")).split())
        missing = self._required_scopes - scopes
        if missing:
            raise AuthenticationError("token lacks required scopes", status_code=403)
        subject = claims["sub"]
        tenant_id = claims.get(self._tenant_claim) or subject
        if (
            not isinstance(subject, str)
            or not subject
            or not isinstance(tenant_id, str)
            or not tenant_id
        ):
            raise AuthenticationError("token identity claims are invalid")
        return Principal(subject=subject, tenant_id=tenant_id, scopes=scopes)

    async def _key_for(self, kid: str) -> Any:
        if time.monotonic() >= self._keys_expire_at:
            await self._refresh_keys()
        if kid not in self._keys:
            await self._refresh_keys(force=True)
        key = self._keys.get(kid)
        if key is None:
            raise AuthenticationError("token signing key is unknown")
        return key

    async def _refresh_keys(self, *, force: bool = False) -> None:
        async with self._lock:
            if not force and time.monotonic() < self._keys_expire_at and self._keys:
                return
            owns_client = self._client is None
            client = self._client or httpx.AsyncClient(timeout=10, follow_redirects=False)
            try:
                response = await client.get(f"{self._issuer}/.well-known/jwks.json")
                response.raise_for_status()
                document = response.json()
                if not isinstance(document, dict):
                    raise AuthenticationError("Cognito JWKS response is invalid")
                keys = document.get("keys")
                if not isinstance(keys, list):
                    raise AuthenticationError("Cognito JWKS response is invalid")
                parsed: dict[str, Any] = {}
                for item in keys:
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("kid"), str)
                        and item.get("alg", "RS256") == "RS256"
                    ):
                        parsed[item["kid"]] = jwt.PyJWK.from_dict(item).key
                if not parsed:
                    raise AuthenticationError("Cognito JWKS contains no usable keys")
                self._keys = parsed
                self._keys_expire_at = time.monotonic() + self._cache_seconds
            except AuthenticationError:
                raise
            except (httpx.HTTPError, KeyError, ValueError, jwt.PyJWTError) as exc:
                raise AuthenticationError("unable to retrieve Cognito signing keys") from exc
            finally:
                if owns_client:
                    await client.aclose()


def build_authenticator(settings: AppSettings) -> RequestAuthenticator:
    if settings.auth_mode == "development":
        return DevelopmentAuthenticator(scopes=settings.required_scopes)
    if settings.auth_mode != "cognito":
        raise ValueError("AUTH_MODE must be either development or cognito")
    if not settings.cognito_issuer or not settings.cognito_client_id:
        raise ValueError("Cognito authentication requires issuer and client ID")
    return CognitoAuthenticator(
        issuer=settings.cognito_issuer,
        client_id=settings.cognito_client_id,
        required_scopes=settings.required_scopes,
        tenant_claim=settings.cognito_tenant_claim,
        jwks_cache_seconds=settings.cognito_jwks_cache_seconds,
    )


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise AuthenticationError("missing bearer token")
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.casefold() != "bearer" or not token.strip():
        raise AuthenticationError("invalid authorization header")
    return token.strip()
