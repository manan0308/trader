from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from broker.groww_client import generate_totp_code
from runtime.groww_auth_refresh import render_runtime_env


def test_generate_totp_code_matches_rfc_vector() -> None:
    # RFC 6238 shared secret "12345678901234567890" in base32 form.
    secret = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
    assert generate_totp_code(secret, digits=8, for_time=59) == "94287082"
    assert generate_totp_code(secret, digits=6, for_time=59) == "287082"


def test_render_runtime_env_replaces_managed_keys_only() -> None:
    rendered = render_runtime_env(
        {
            "GROWW_ACCESS_TOKEN": "old-token",
            "GROWW_ACCESS_TOKEN_SOURCE": "manual",
            "KEEP_ME": "yes",
        },
        {
            "GROWW_ACCESS_TOKEN": "new-token",
            "GROWW_ACCESS_TOKEN_SOURCE": "api_key_totp",
            "GROWW_LAST_PUBLIC_IP": "89.167.33.51",
        },
    )
    assert 'KEEP_ME="yes"' in rendered
    assert 'GROWW_ACCESS_TOKEN="new-token"' in rendered
    assert 'GROWW_ACCESS_TOKEN_SOURCE="api_key_totp"' in rendered
    assert 'GROWW_LAST_PUBLIC_IP="89.167.33.51"' in rendered
    assert 'old-token' not in rendered
