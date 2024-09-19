import subprocess
from os import listdir
import pandas as pd

import os
RUN = False
SPECIFIC_INSTANCES = True
POLICIES = True
Direc = "instances"
files = os.listdir(Direc)
# Filtering only the files.
names = [f[0:-5] for f in files if os.path.isfile(Direc+'/'+f)]
names = [n for n in names if n[-4:] != "mean"]
names = sorted(names, reverse=True)

if SPECIFIC_INSTANCES:
    names = ['I2_N5_T30_C100_0', 'I2_N5_T30_C150_0', 'I2_N5_T30_C200_0', 'I2_N5_T100_C100_0', 'I2_N5_T100_C150_0', 'I2_N5_T100_C200_0',
    'I2_N7_T30_C140_0', 'I2_N7_T30_C210_0', 'I2_N7_T30_C280_0','I2_N7_T100_C140_0', 'I2_N7_T100_C210_0', 'I2_N7_T100_C280_0']
    names = ['I2_N10_T30_C400_0','I2_N10_T30_C350_0','I2_N10_T30_C325_0','I2_N10_T30_C300_0','I2_N10_T30_C275_0','I2_N10_T100_C400_0',
              'I2_N10_T100_C350_0', 'I2_N10_T100_C325_0', 'I2_N10_T100_C300_0', 'I2_N10_T100_C275_0']
if POLICIES:
    nplus = []
    for n in names:
        nplus.append(n)
        for p in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            nplus.append(n + "_" + p)
    names = nplus

print(*names, sep="\n")

if RUN:
    for name in names:
        subprocess.run(['python', 'main.py', name])

# aggregate
df = pd.DataFrame(columns=['name', 'objVal', 'runTime', 'gap', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'])

for name in names:
    nameres = name + "_res.xlsx"
    path ='results/'
    dfres = pd.read_excel(path + nameres, sheet_name='general', index_col='param')

    nInst = name #dfres.loc['Inst name']['value']
    objValue = dfres.loc['objValue']['value']
    runTime = dfres.loc['runtime']['value']
    gap = dfres.loc['gap']['value']
    Z1 = dfres.loc['Z1']['value']
    Z2 = dfres.loc['Z2']['value']
    Z3 = dfres.loc['Z3']['value']
    Z4 = dfres.loc['Z4']['value']
    Z5 = dfres.loc['Z5']['value']

    df.loc[len(df)] = [nInst, objValue, runTime, gap, Z1, Z2, Z3, Z4, Z5]
df.to_excel('table_results_policies.xlsx')
