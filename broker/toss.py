"""Toss Securities Open API — read-only client (잔고/보유 조회 전용).

HARD CONSTRAINT: no order/modify/cancel capability may exist in this module
until the Tier-2 paper gate (~2026-08-29) passes — and then only behind the
safety checklist in REVIEW_2026-06-12_auto_trading.md. A regression test
(tests/broker/test_toss_readonly.py) asserts the client exposes no trading
surface.

Auth is OAuth 2.0 Client Credentials; keys are issued personally in WTS(PC웹)
설정 > Open API. Keys live ONLY on the ubuntu server's /etc/recostock.env —
never in GitHub secrets (the Actions pipeline consumes the sanitized snapshot
file instead, see scripts/sync_broker_holdings.py).

Endpoint paths are PROVISIONAL. The Open API launched 2026-06 and is rolling
out gradually; the account/positions paths below are best guesses from the
published docs (https://openapi.tossinvest.com/openapi-docs/overview.md) and
MUST be confirmed against a real key. They are env-overridable so the fix
needs no code change:

    TOSS_API_BASE        default https://openapi.tossinvest.com
    TOSS_TOKEN_PATH      default /oauth2/token
    TOSS_POSITIONS_PATH  default /api/v1/account/positions
    TOSS_BALANCE_PATH    default /api/v1/account/balance
    TOSS_APP_KEY / TOSS_APP_SECRET   credentials (required)

Rate limits are generous for this use (1 call/day); no retry sophistication
needed beyond a simple timeout.
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)

DEFAULT_BASE = "https://openapi.tossinvest.com"
DEFAULT_TOKEN_PATH = "/oauth2/token"
DEFAULT_POSITIONS_PATH = "/api/v1/account/positions"
DEFAULT_BALANCE_PATH = "/api/v1/account/balance"

# Tolerant response-key candidates — the exact schema is unconfirmed until the
# key is approved, so normalization tries each alias in order.
_TICKER_KEYS = ("ticker", "symbol", "stockCode", "code")
_QTY_KEYS = ("quantity", "qty", "holdingQuantity", "balanceQty")
_VALUE_KEYS = ("evaluationAmount", "evalAmount", "marketValue", "value", "amount")
_CASH_KEYS = ("withdrawableAmount", "cashBalance", "depositAmount", "cash")
_LIST_KEYS = ("positions", "data", "result", "items")


class TossReadOnlyClient:
    """Minimal OAuth2 client-credentials GET-only client."""

    def __init__(self, app_key: str, app_secret: str,
                 base_url: str = DEFAULT_BASE, session=None, timeout: int = 15):
        if not app_key or not app_secret:
            raise ValueError("TOSS_APP_KEY / TOSS_APP_SECRET required")
        self._key, self._secret = app_key, app_secret
        self.base_url = base_url.rstrip("/")
        if session is None:
            import requests
            session = requests.Session()
        self._session = session
        self._timeout = timeout
        self._tok: str | None = None
        self._tok_exp: float = 0.0
        self.token_path = os.environ.get("TOSS_TOKEN_PATH", DEFAULT_TOKEN_PATH)
        self.positions_path = os.environ.get("TOSS_POSITIONS_PATH", DEFAULT_POSITIONS_PATH)
        self.balance_path = os.environ.get("TOSS_BALANCE_PATH", DEFAULT_BALANCE_PATH)

    @classmethod
    def from_env(cls) -> "TossReadOnlyClient":
        return cls(
            app_key=os.environ.get("TOSS_APP_KEY", ""),
            app_secret=os.environ.get("TOSS_APP_SECRET", ""),
            base_url=os.environ.get("TOSS_API_BASE", DEFAULT_BASE),
        )

    # ── auth ──────────────────────────────────────────────────────────────────
    def _token(self) -> str:
        if self._tok and time.monotonic() < self._tok_exp - 60:
            return self._tok
        # Standard OAuth2 client-credentials form. If Toss uses bespoke field
        # names (e.g. KIS-style appkey/appsecret JSON), adjust here once the
        # real response is observable.
        resp = self._session.post(
            self.base_url + self.token_path,
            data={"grant_type": "client_credentials",
                  "client_id": self._key, "client_secret": self._secret},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        tok = body.get("access_token") or body.get("accessToken")
        if not tok:
            raise RuntimeError(f"token response missing access_token: keys={list(body)}")
        ttl = float(body.get("expires_in") or body.get("expiresIn") or 300)
        self._tok, self._tok_exp = tok, time.monotonic() + ttl
        return tok

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self._session.get(
            self.base_url + path, params=params or {},
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ── read-only queries ─────────────────────────────────────────────────────
    def positions(self) -> list[dict]:
        """Normalized holdings: [{ticker, qty, value}] in account currency."""
        return normalize_positions(self._get(self.positions_path))

    def cash_value(self) -> float:
        """Available cash in account currency. 0.0 when not resolvable."""
        body = self._get(self.balance_path)
        flat = body.get("data") if isinstance(body.get("data"), dict) else body
        for k in _CASH_KEYS:
            v = flat.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        log.warning("Toss balance: no cash field among %s — keys=%s", _CASH_KEYS, list(flat))
        return 0.0


def normalize_positions(raw) -> list[dict]:
    """Schema-tolerant extraction of [{ticker, qty, value}] from the response.

    Accepts a bare list or a dict wrapping one under positions/data/result/
    items. Rows without a resolvable ticker or a positive value are skipped
    (logged) rather than raising — a partial snapshot with a warning beats a
    dead sync.
    """
    items = raw
    if isinstance(raw, dict):
        for k in _LIST_KEYS:
            if isinstance(raw.get(k), list):
                items = raw[k]
                break
        else:
            items = []
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        tk = next((str(it[k]).upper() for k in _TICKER_KEYS if it.get(k)), None)
        val = next((it[k] for k in _VALUE_KEYS if it.get(k) is not None), None)
        qty = next((it[k] for k in _QTY_KEYS if it.get(k) is not None), None)
        try:
            val_f = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            val_f = 0.0
        if not tk or val_f <= 0:
            log.warning("Toss positions: skipping unparseable row %s", it)
            continue
        out.append({"ticker": tk, "qty": float(qty) if qty is not None else None,
                    "value": val_f})
    return out


def holdings_to_weights(positions: list[dict], cash_value: float) -> tuple[dict, float]:
    """Value-based capital fractions: ({ticker: weight}, cash_weight).

    All inputs must share one currency (whatever the account reports in —
    fractions are currency-agnostic as long as it's consistent).
    """
    total = sum(p["value"] for p in positions) + max(0.0, float(cash_value))
    if total <= 0:
        return {}, 0.0
    weights = {}
    for p in positions:
        weights[p["ticker"]] = weights.get(p["ticker"], 0.0) + p["value"] / total
    weights = {k: round(v, 4) for k, v in weights.items()}
    return weights, round(max(0.0, float(cash_value)) / total, 4)


def build_snapshot(positions: list[dict], cash_value: float, as_of: str) -> dict:
    """Sanitized snapshot for data/broker/holdings.json — weights ONLY.

    No quantities, no amounts, no account identifiers: docs/ is published to
    GitHub Pages and the repo may be public, so absolute sizes never leave the
    server. Fractions reveal allocation, which the daily report already does.
    """
    weights, cash_w = holdings_to_weights(positions, cash_value)
    return {"asOf": str(as_of), "source": "toss-openapi",
            "weights": weights, "cashWeight": cash_w}
