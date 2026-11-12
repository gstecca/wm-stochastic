"""Recompute the results table (Table 5 in the submitted paper, Table 6 in R1)
from Instances_list.xlsx, in the original way (zeros excluded) and the corrected
way (all instances with a feasible solution). Usage: python recompute_table6.py"""
import pandas as pd

d = pd.read_excel('Instances_list.xlsx')
d['N'] = d['name'].str.split('_').str[1].str[1:].astype(int)
d['C'] = d['name'].str.split('_').str[4].str[1:].astype(int)
s = d[d['solved'] == True]
pos = lambda x: x[x > 0].mean()  # aggregation used in the original notebook

orig = d.groupby(['N', 'C']).agg(
    solved=('solved', 'sum'), VSS_a=('VSS', pos), VSS_M=('VSS', 'max'),
    time_a=('runTime', pos), time_M=('runTime', 'max'), gap_a=('gap', pos), gap_M=('gap', 'max'),
    timeEEV_a=('runTimeEEV', pos), timeEEV_M=('runTimeEEV', 'max'),
    gapEEV_a=('gapEEV', pos), gapEEV_M=('gapEEV', 'max'))
corr = s.groupby(['N', 'C']).agg(
    solved=('solved', 'size'), VSS_pos=('VSS', lambda x: (x > 1e-6).sum()),
    VSS_a=('VSS', 'mean'), VSS_M=('VSS', 'max'),
    time_a=('runTime', 'mean'), time_M=('runTime', 'max'), gap_a=('gap', 'mean'), gap_M=('gap', 'max'),
    timeEEV_a=('runTimeEEV', 'mean'), timeEEV_M=('runTimeEEV', 'max'),
    gapEEV_a=('gapEEV', 'mean'), gapEEV_M=('gapEEV', 'max'))

pd.set_option('display.width', 200)
print('ORIGINAL (submitted Table 5; averages over strictly positive values only)')
print(orig.round(2).to_string())
w = (orig['solved'] * orig['VSS_a']).sum() / orig['solved'].sum()
print(f'weighted mean of the original group averages (Reviewer 1): {w:.2f}\n')
print('CORRECTED (R1 Table 6; averages over all instances with a feasible solution; gaps are fractions)')
print(corr.round(2).to_string())
opt = s[s['gap'] <= 1e-4]
print(f"\nmean VSS over {len(s)} feasible instances: {s['VSS'].mean():.2f}")
print(f"mean VSS over {len(opt)} instances with gap <= 1e-4: {opt['VSS'].mean():.2f}")
print(f"instances with VSS > 1e-6: {(s['VSS'] > 1e-6).sum()}, mean over those: {s.loc[s['VSS'] > 1e-6, 'VSS'].mean():.2f}")
print(f"median VSS: {s['VSS'].median():.2f}")
