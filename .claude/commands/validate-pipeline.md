Run the data pipeline integrity checks for this project.

1. Run the look-ahead bias test suite:
   ```
   pytest tests/data/test_lookahead.py -v
   ```

2. Verify `config.py` trading cost constants are not lower than Toss Securities actual rates:
   - `COMMISSION_ROUNDTRIP` must be >= 0.002 (0.2%)
   - `TOTAL_COST_ROUNDTRIP` must be >= 0.0025

3. Check that all factor functions in `features/factors.py` use only `.rolling()`, `.shift()`, `.diff()`, or other strictly look-back operations. Flag any `.iloc[future]`, sorted-index trickery, or future-bar references.

4. If $ARGUMENTS is provided, also check that specific file for look-ahead issues.

Report: list any violations found, or confirm "No look-ahead bias or cost underestimation found."
