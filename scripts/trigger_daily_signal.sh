#!/usr/bin/env bash
# Trigger the Daily ETF Signal workflow via GitHub workflow_dispatch.
#
# GitHub Actions' `schedule` trigger is best-effort and routinely delayed
# 1-3h (or skipped). This external trigger fires the workflow on time via the
# REST API, which queues within seconds.
#
# Setup (Ubuntu):
#   1. Create a GitHub token with workflow-dispatch rights:
#        - Fine-grained PAT: repo wansoo88/recostock, Actions = Read and write
#        - or classic PAT with `workflow` scope
#   2. Store it (NOT in git):  echo 'GH_DISPATCH_TOKEN=ghp_xxx' | sudo tee /etc/recostock.env
#                              sudo chmod 600 /etc/recostock.env
#   3. crontab -e  (assuming server clock is UTC):
#        0 13 * * 1-5  . /etc/recostock.env && /path/to/trigger_daily_signal.sh >> /var/log/recostock_trigger.log 2>&1
#      (13:00 UTC = 22:00 KST. If server is on KST, use `0 22 * * 1-5`.)
set -euo pipefail

REPO="${RECOSTOCK_REPO:-wansoo88/recostock}"
WORKFLOW="${RECOSTOCK_WORKFLOW:-daily_signal.yml}"
REF="${RECOSTOCK_REF:-main}"

if [[ -z "${GH_DISPATCH_TOKEN:-}" ]]; then
  echo "[$(date -u +%FT%TZ)] ERROR: GH_DISPATCH_TOKEN not set" >&2
  exit 1
fi

http_code=$(curl -sS -o /tmp/recostock_dispatch_resp -w '%{http_code}' \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GH_DISPATCH_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches" \
  -d "{\"ref\":\"${REF}\"}")

if [[ "${http_code}" == "204" ]]; then
  echo "[$(date -u +%FT%TZ)] OK: dispatched ${WORKFLOW} on ${REF} (HTTP 204)"
else
  echo "[$(date -u +%FT%TZ)] ERROR: HTTP ${http_code}" >&2
  cat /tmp/recostock_dispatch_resp >&2
  exit 1
fi
