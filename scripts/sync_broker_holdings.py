"""Fetch real holdings from Toss Open API (read-only) → data/broker/holdings.json.

Runs on the UBUNTU SERVER (the only place the API keys exist), right before
trigger_daily_signal.sh fires the Actions pipeline; the wrapping cron commits
the sanitized snapshot so Actions consumes a file, never a key. The snapshot
contains capital-fraction weights only — no quantities, amounts, or account
identifiers (repo/Pages may be public).

Setup (Ubuntu):
  1. WTS(PC웹) > 설정 > Open API에서 키 발급 후:
       echo 'TOSS_APP_KEY=...'    | sudo tee -a /etc/recostock.env
       echo 'TOSS_APP_SECRET=...' | sudo tee -a /etc/recostock.env
       sudo chmod 600 /etc/recostock.env
  2. crontab -e (UTC clock) — sync ~10 min before the 13:00 dispatch:
       50 12 * * 1-5  . /etc/recostock.env && cd /path/to/recostock \
         && python scripts/sync_broker_holdings.py \
         && git add data/broker/holdings.json \
         && git commit -m "broker: holdings snapshot [skip ci]" && git push

Failure mode is graceful: if this never runs (key not yet approved, server
down), the pipeline silently falls back to the tracker assumption — exactly
the pre-integration behavior.

Usage:
    python scripts/sync_broker_holdings.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from broker.toss import TossReadOnlyClient, build_snapshot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sync_broker_holdings")

OUT_PATH = Path("data/broker/holdings.json")


def main(dry_run: bool = False) -> int:
    client = TossReadOnlyClient.from_env()
    positions = client.positions()
    cash = client.cash_value()
    snap = build_snapshot(positions, cash,
                          as_of=datetime.now(timezone.utc).date().isoformat())
    if not snap["weights"] and snap["cashWeight"] <= 0:
        log.error("Empty snapshot (no positions, no cash) — refusing to overwrite")
        return 1
    body = json.dumps(snap, ensure_ascii=False, indent=2) + "\n"
    if dry_run:
        print(body)
        return 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(body, encoding="utf-8")
    log.info("Wrote %s: %d position(s), cash %.1f%%",
             OUT_PATH, len(snap["weights"]), snap["cashWeight"] * 100)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the snapshot instead of writing it")
    sys.exit(main(dry_run=ap.parse_args().dry_run))
