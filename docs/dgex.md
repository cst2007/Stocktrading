# dGEX/dSpot calculation

The `dGEX/dSpot` metric estimates how net gamma exposure (GEX) changes with respect to the underlying spot price. It is computed in `barchart/derived_metrics.py` using a central finite-difference approximation:

1. **Determine a spot bump size.** The code uses `max(10.0, spot * 0.0025)` so that the bump scales with the underlying level but never falls below 10 points. This sets the distance between the up and down spot shocks used in the derivative estimate.
2. **Limit strikes evaluated.** Strikes are sorted and a window of up to ±15 strikes around the strike closest to spot is selected. This focuses the calculation on the most relevant part of the surface and avoids unnecessary work.
3. **Compute net GEX at bumped spot levels.** For each strike in the window, net GEX is recomputed twice using the shocked spot levels `spot + bump_size` and `spot - bump_size`. Net GEX multiplies per-contract gamma by open interest and by the spot-squared contract multiplier (100):
   
   ```python
   spot_term = spot_level ** 2 * 100
   call_component = sum(call_gamma * call_open_interest * spot_term)
   put_component = sum(put_gamma * put_open_interest * spot_term)
   net_gex = call_component - put_component
   ```
4. **Central difference derivative.** The directional sensitivity is the slope between the up- and down-shocked net GEX values:
   
   ```python
   derivative = (net_gex_up - net_gex_down) / (2 * bump_size)
   ```
   The result is rounded to two decimals and stored per strike as `dGEX/dSpot`.
5. **Ranking.** The top five positive slopes receive `Top N` labels and the most negative (excluding ties with the top list) receive `Bottom N` labels in the `dGEX/dSpot Rank` column.

This approach provides a smooth numerical estimate of how dealer gamma exposure would change for small spot moves around each strike, highlighting areas where the gamma profile is steepest.
