# Summary of work for Giuseppe's points (CAIE-D-26-05414, R1)

The answers are in `R1/response_giuseppe.tex`, with a compiled `R1/response_giuseppe.pdf`. `R1/overleaf/` is empty, so the work is based on the manuscript PDF; the manuscript changes are given as blue text to paste in.

## 1. `wm-collection-stochastic/` (about 70 MB)

- **`instances/`**: the 720 instances plus their 720 expected-value (`_mean`) versions, the exact files used in the paper.
- **`results/`**: the per-instance solutions — 597 each for the stochastic model, the expected-value model and its evaluation on the scenarios, plus 2136 policy runs (178 instances × 12 policies).
- **`summary/`**: CSV files with one row per instance, per instance–policy pair, the ML dataset, the features and the collection-point types.
- **`code/generate_instances.py`**: a standalone generator taken from the notebook. All 1440 files were regenerated and compared sheet by sheet: no differences, and the ML features match too.
- **`code/compute_table5.py`**: rebuilds the results table (Table 5 in the submitted version, Table 6 in R1) both the old way and the corrected way. The code folder also has `main.py`, `myutils.py` and the instance lists.
- **`README.md`**: covers file format, units, generation procedure, seeds, the model constants, and how to run the code.

Published at https://github.com/gstecca/wm-collection-stochastic under the Apache 2.0 license (the LICENSE file that came with the GitHub repository).

## 2. The answers

**R1.3 (M and Qmax)**
- All instances use M = Qmax = 10⁴, and that value is valid everywhere.
- **Bounds for M (constraints 10–13):** M ≥ Q̄ is enough, where Q̄ is the largest total waste in any scenario. Q̄ is at most 1024 in the actual instances (255·n in general).
- **Bound for Qmax (17–18):** Qmax ≥ max{C, Q̄ − C} is enough; that is at most 874 here.
- **Constraint (7):** the code actually uses Tmax as the constant, not M. The draft includes a short proof that Tmax is valid there, because travel times are Euclidean and service times are zero.
- **No numerical problem:** the data are integers or multiples of 0.5, so a load that differs from C always clears Gurobi's tolerance.
- **Empirical check with the tightest constants (run 2026-11-12 on labopt, Gurobi 13.0.1, 4 parallel jobs × 2 threads each):**
  - **Sample:** all 297 near-optimal instances with n=5 and n=6, plus 20 with n=7 and 15 with n=8 (random, seed 1), solved with the stochastic model. The 297 small ones were also solved with the expected-value model. That makes 629 pairs, each solved with M = Qmax = 10⁴ and with the tight values M = Q̄ and Qmax = max{C, Q̄ − C}, which range from 253.5 to 1024.
  - **Same optima:** all 629 pairs are optimal with both settings. The objective values agree to within 10⁻⁶ relative in 627 pairs. The other two (EM, `I3_N6_M3_T45_C150_DepCentral_s1/s3`) differ by less than 0.01 in absolute value, which is solver tolerance. Routes differ in most pairs, but they are alternative optima with the same cost.
  - **Slightly faster:** with the tight values, total time drops from 13160 s to 11670 s (−11%) and branch-and-bound nodes by 23%. The largest effect is on n=7, with a geometric-mean runtime ratio of 0.88.
  - **Where it is:** the code is in `R1/qmax_test/` (`run_qmax_test.py`, `analyze.py`, `jobs.txt`, `run_all.sh`), the per-pair results are in `R1/qmax_test/results/comparison.csv`, and the raw files and Gurobi logs are on labopt in `~/qmaxtest/`.
  - **Added to the draft:** a new paragraph in the R1.3 answer and one sentence in the manuscript text.
- **When the load equals capacity exactly:** the constraints leave the saturation variable free, but the objective never loses by treating it as saturated. Checked in all 597 solutions: it happens 137 times, and whenever another collection point followed (32 cases) the solver marked it saturated and skipped the rest. It was left unsaturated only when the point was last on its route (42 cases), where the cost is identical either way. The draft also gives an optional reformulation that fixes the convention in the constraints themselves.

**R1.4 (VSS)**
- The reviewer is right: weighting the group averages of the results table (Table 5 in the submitted version, now Table 6) gives 27.02%, reproduced exactly.
- **First error:** the group averages left out instances with VSS = 0 (the analysis kept only positive values).
- **Second error:** 21.33% is the average over the 548 instances with gap ≤ 0.1%, not over the 597 the text says.
- **Corrected figures:** 19.65% over the 597 instances, 21.33% over the 548 near-optimal ones. VSS is positive in 356 instances (32.96% average over those), and the median is 0.12%.
- **What doesn't change:** VSS still increases with capacity. The draft includes the revised Table 6, new Section 5.2 text and new Conclusions wording.
- **Table numbering:** your co-author's new literature table (`tab:literature`, sidewaystable) is now Table 1, so the old Table 5 is **Table 6**, and the notation table is Table 3 instead of 2. `response_giuseppe.tex` now uses the new numbers. The reviewer's quoted comment still says "Table 5", and the first answer sentence adds "(Table 6 in the revised manuscript)".

**Where the R1.4 numbers come from (all paths relative to the repo root `wm-stochastic/`)**
- **Input data:** `analysis/Instances_list.xlsx`, one row per instance (720 rows), with columns `solved`, `objVal`, `runTime`, `gap`, `EVV`, `runTimeEEV`, `gapEEV`, `VSS`.
  - Not to be confused with `Instances_list.xlsx` at the repo root, an older version (January 2025) that has no `runTimeEEV` column, so it cannot be the source of the table.
  - `Instances_list_548.xlsx` at the root has the same values plus the policy columns (`FILTRO`, `bestpolicy`, `policyvalue`) and gives the same 548 instances and 21.33%.
- **Original Table 5 (submitted version):** `analysis/analisi_risultati_totali.ipynb`.
  - Cell 4 groups by `['N','C']` and writes `analysis/Instances_list_agg_NC.xlsx`, the table in the paper.
  - The bug is there: `VSS_mean`, `gap_mean`, `runTime_mean` and the EEV columns use `lambda x: x[x > 0].mean()`, so zeros are excluded from the averages.
  - Cell 4 of the same notebook, run with `groupBY = ['N']` or `['C']`, gives `Instances_list_agg_N.xlsx` and `Instances_list_agg_C.xlsx`.
- **The 21.33%:** same notebook, cells 8 and 11: `dftot[(solved==True) & (gap<=0.1)]['VSS'].mean()`, which is 548 instances. The same code is in `analysis/analisi_N8.ipynb`, cells 6 and 9.
  - Note that `gap<=0.1` means 10%, not 0.1%, because gaps are fractions.
  - It makes no difference: no gap falls between 10⁻⁴ and 0.22, so thresholds of 10⁻⁴, 10⁻³ and 0.1 all select the same 548 instances.
- **Corrected figures:** `wm-collection-stochastic/code/compute_table5.py` in the public repository, which reads `summary/` there.
  - That folder isn't in this local clone. The figures can also be recomputed directly from `analysis/Instances_list.xlsx`, and I re-checked them today (2026-11-12):
    - Old-style weighted mean: **27.02**.
    - Mean over the 597 solved instances: **19.65**.
    - Mean over the 548 with gap ≤ 10⁻⁴: **21.33**.
    - Median: **0.12**.
    - Group values: for example n=7, C=250 gives 28.20, and n=8, C=150 gives VSS 0.15 with gap 0.47.
  - **"VSS positive in 356 instances, average 32.96%" uses the threshold VSS > 10⁻⁶.** With a strict `VSS > 0` you get 413 instances averaging 28.41%, because 57 instances have VSS of about 10⁻¹² (floating-point noise). If the text says "positive", it should say "VSS > 10⁻⁶" or "VSS ≥ 0.0001%".

**R2.1 (Giuseppe's part):** the core argument is that in our model the amount collected at a point depends on when the vehicle arrives, which depends on the route order. In day-to-day models the vehicle load depends only on which points are on the route, not their order. The draft gives three consequences of this and explains why simply splitting the day into finer periods doesn't capture it. It also proposes two new Table 1 columns.

**R3 (repository):** a description of the repository, plus a data availability statement.

## Things to raise with co-authors (also marked as internal notes in the draft)

1. **The "not solved" instances are actually proven infeasible.** Gurobi reports all 123 as infeasible; none simply ran out of time. In 120 of them (corner depot, Tmax = 15) the round trip to the farthest point is longer than Tmax. The other 3 are `I3_N6_M3_T15_C{150,200,250}_DepCentral_s1`. Section 5.2 describes them wrongly, and this answers one of Simona's Reviewer 3 points.
2. **Section 5.1 doesn't match the generator.**
   - Initial demand is uniform on [10, 30] for every type; the type multiplier is commented out in the notebook.
   - The increments are uniform on [0, 25], [25, 50] and [50, 75], not uniform [10, 30] scaled by 0.5 or 1.5.
   - A corrected paragraph for Simona is in the draft.
3. **The 720 instances come from only 20 underlying layouts.** Because the seed is reset for each instance, the same n and seed always produce the same point locations and demands, across all 36 combinations of fleet size, capacity, time horizon and depot. The README states this openly, and it's relevant to the reviewers' concerns about generalisation.
4. **Gaps in Table 6 (old Table 5) are fractions, not percentages.** 0.68 means 68%. The old gap averages also left out zero values; the revised table fixes both.
5. **Francesca should check the Table 1 classifications** for the other papers. They are based only on how Section 2.2 describes those papers.
6. **New items seen in the R1 overleaf (`R1/overleaf/rev_CAIE_main.tex`):**
   - **Duplicate labels:** the new comparison table (line ~298) and the old synthesis table (line ~470) both use `\label{tab:literature}`, so the `\ref` in the text resolves to only one of them. One needs renaming.
   - **A request for you:** Simona's note `\sm{...}` (line ~444) asks you to check the new table and add a row for Huang (2026).
   - **Contradiction to settle with Simona:** the new `\rev{}` paragraph after Table 6 says the "not solved" instances should be read "rather than as evidence of infeasibility". That contradicts point 1: all 123 are proven infeasible by Gurobi.

## Files (in `/mnt/c/Users/STECCAGIUSEPPE/git/wm-stochastic`; `wm-collection-stochastic/` is in the GitHub repository, not in this clone)

- `R1/response_giuseppe.tex`
- `R1/response_giuseppe.pdf`
- `wm-collection-stochastic/README.md`
- `wm-collection-stochastic/code/compute_table5.py`
- `wm-collection-stochastic/code/generate_instances.py`
