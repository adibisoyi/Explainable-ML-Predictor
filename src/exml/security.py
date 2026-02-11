from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class Principal:
    role: str
    key_id: str


def load_api_keys() -> dict[str, Principal]:
    raw = os.getenv("EXML_API_KEYS")
    if raw:
        mapping: dict[str, dict[str, str]] = json.loads(raw)
    else:
        env = os.getenv("EXML_ENV", "dev").lower()
        if env == "prod":
            raise RuntimeError("EXML_API_KEYS must be set when EXML_ENV=prod")
        mapping = {
            "dev-admin-key": {"role": "admin", "key_id": "local-dev"},
            "dev-predict-key": {"role": "predictor", "key_id": "local-predictor"},
        }

    keys: dict[str, Principal] = {}
    for key, data in mapping.items():
        role = data.get("role")
        if role not in {"admin", "predictor"}:
            raise RuntimeError(f"Invalid role configured for API key: {role}")
        keys[key] = Principal(role=role, key_id=data.get("key_id", key[-6:]))
    return keys


def authorize_request(request: Request, allowed_roles: set[str]) -> Principal:
    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")

    app_keys: Any = request.app.state.api_keys
    if not isinstance(app_keys, dict):
        raise HTTPException(status_code=500, detail="Invalid server auth configuration")

    principal = app_keys.get(api_key)
    if principal is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not isinstance(principal, Principal):
        raise HTTPException(status_code=500, detail="Invalid server auth configuration")

    if principal.role not in allowed_roles:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    request.state.principal = f"{principal.key_id}:{principal.role}"
    return principal
