"""Broker integration — READ-ONLY until the Tier-2 gate (~2026-08-29).

This package may only ever OBSERVE the real account (balance/positions/fills).
Order placement, modification, and cancellation are structurally absent by
design — adding them before the Tier-2 paper gate passes violates the
project's two-stage gate principle (CLAUDE.md 부록 B, 원칙 4). See
REVIEW_2026-06-12_auto_trading.md for the staged roadmap and the safety
checklist that must precede any write capability.
"""
