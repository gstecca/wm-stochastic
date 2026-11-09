# Summary of work for Giuseppe's points (CAIE-D-26-05414, R1)

The answers are in `R1/response_giuseppe.tex`, with a compiled `R1/response_giuseppe.pdf`. `R1/overleaf/` is empty, so the work is based on the manuscript PDF; the manuscript changes are given as blue text to paste in.

## 1. `wm-collection-stochastic/` (about 70 MB)

- **`instances/`**: the 720 instances plus their 720 expected-value (`_mean`) versions, the exact files used in the paper.
- **`results/`**: the per-instance solutions — 597 each for the stochastic model, the expected-value model and its evaluation on the scenarios, plus 2136 policy runs (178 instances × 12 policies).
- **`summary/`**: CSV files with one row per instance, per instance–policy pair, the ML dataset, the features and the collection-point types.
- **`code/generate_instances.py`**: a standalone generator taken from the notebook. All 1440 files were regenerated and compared sheet by sheet: no differences, and the ML features match too.
- **`code/compute_table5.py`**: rebuilds Table 5 both the old way and the corrected way. The code folder also has `main.py`, `myutils.py` and the instance lists.
- **`README.md`**: covers file format, units, generation procedure, seeds, the model constants, and how to run the code.

Published at https://github.com/gstecca/wm-collection-stochastic under the Apache 2.0 license (the LICENSE file that came with the GitHub repository).

## 2. The answers

**R1.3 (M and Qmax)**
- All instances use M = Qmax = 10⁴, and that value is valid everywhere.
- **Bounds for M (constraints 10–13):** M ≥ Q̄ is enough, where Q̄ is the largest total waste in any scenario. Q̄ is at most 1024 in the actual instances (255·n in general).
- **Bound for Qmax (17–18):** Qmax ≥ max{C, Q̄ − C} is enough; that is at most 874 here.
- **Constraint (7):** the code actually uses Tmax as the constant, not M. The draft includes a short proof that Tmax is valid there, because travel times are Euclidean and service times are zero.
- **No numerical problem:** the data are integers or multiples of 0.5, so a load that differs from C always clears Gurobi's tolerance.
- **When the load equals capacity exactly:** the constraints leave the saturation variable free, but the objective never loses by treating it as saturated. Checked in all 597 solutions: it happens 137 times, and whenever another collection point followed (32 cases) the solver marked it saturated and skipped the rest. It was left unsaturated only when the point was last on its route (42 cases), where the cost is identical either way. The draft also gives an optional reformulation that fixes the convention in the constraints themselves.

**R1.4 (VSS)**
- The reviewer is right: weighting the Table 5 group averages gives 27.02%, reproduced exactly.
- **First error:** the group averages left out instances with VSS = 0 (the analysis kept only positive values).
- **Second error:** 21.33% is the average over the 548 instances with gap ≤ 0.1%, not over the 597 the text says.
- **Corrected figures:** 19.65% over the 597 instances, 21.33% over the 548 near-optimal ones. VSS is positive in 356 instances (32.96% average over those), and the median is 0.12%.
- **What doesn't change:** VSS still increases with capacity. The draft includes the revised Table 5, new Section 5.2 text and new Conclusions wording.

**R2.1 (Giuseppe's part):** the core argument is that in our model the amount collected at a point depends on when the vehicle arrives, which depends on the route order. In day-to-day models the vehicle load depends only on which points are on the route, not their order. The draft gives three consequences of this and explains why simply splitting the day into finer periods doesn't capture it. It also proposes two new Table 1 columns.

**R3 (repository):** a description of the repository, plus a data availability statement.

## Things to raise with co-authors (also marked as internal notes in the draft)

1. **The "not solved" instances are actually proven infeasible.** Gurobi reports all 123 as infeasible; none simply ran out of time. In 120 of them (corner depot, Tmax = 15) the round trip to the farthest point is longer than Tmax. The other 3 are `I3_N6_M3_T15_C{150,200,250}_DepCentral_s1`. Section 5.2 describes them wrongly, and this answers one of Simona's Reviewer 3 points.
2. **Section 5.1 doesn't match the generator.**
   - Initial demand is uniform on [10, 30] for every type; the type multiplier is commented out in the notebook.
   - The increments are uniform on [0, 25], [25, 50] and [50, 75], not uniform [10, 30] scaled by 0.5 or 1.5.
   - A corrected paragraph for Simona is in the draft.
3. **The 720 instances come from only 20 underlying layouts.** Because the seed is reset for each instance, the same n and seed always produce the same point locations and demands, across all 36 combinations of fleet size, capacity, time horizon and depot. The README states this openly, and it's relevant to the reviewers' concerns about generalisation.
4. **Gaps in Table 5 are fractions, not percentages.** 0.68 means 68%. The old gap averages also left out zero values; the revised table fixes both.
5. **Francesca should check the Table 1 classifications** for the other papers. They are based only on how Section 2.2 describes those papers.

## Files (in `/home/gstecca/git/wm-stochastic`)

- `R1/response_giuseppe.tex`
- `R1/response_giuseppe.pdf`
- `wm-collection-stochastic/README.md`
- `wm-collection-stochastic/code/compute_table5.py`
- `wm-collection-stochastic/code/generate_instances.py`
