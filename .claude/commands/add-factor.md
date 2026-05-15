Add a new technical factor to `features/factors.py` safely.

The factor name or description is: $ARGUMENTS

Steps:
1. Implement the factor in `features/factors.py` using only look-back operations (rolling, shift, diff). Name the column descriptively (e.g., `mom_5d`, `rsi_14`, `vol_ratio`).
2. Run `pytest tests/data/test_lookahead.py -v` and confirm it passes.
3. Add a one-line comment in the function only if the formula is non-obvious.
4. In `features/factors.py:compute_factors()`, add the new column and confirm `df.dropna()` is still called at the end.

Do NOT:
- Use `.shift(-N)` with negative N (future look).
- Use `.fillna(method='bfill')` which can pull future values.
- Hardcode magic numbers — derive from `config.py` if they are thresholds, or use a named variable.

After adding, report: factor name, formula summary, and test result.
