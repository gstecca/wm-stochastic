import subprocess
from os import listdir
import pandas as pd

import os
RUN = False
SPECIFIC_INSTANCES = True
Direc = "instances"
files = os.listdir(Direc)
# Filtering only the files.
names = [f[0:-5] for f in files if os.path.isfile(Direc+'/'+f)]
names = [n for n in names if n[-4:] != "mean"]
names = sorted(names, reverse=True)

if SPECIFIC_INSTANCES:
    #names = ['I2_N10_T30_C275_0', 'I2_N10_T30_C325_0','I2_N10_T100_C275_0', 'I2_N10_T100_C325_0']
    #names = ['I2_N10_T100_C325_0']
    names = ['I2_N7_T30_C100_0', 'I2_N7_T30_C120_0', 'I2_N7_T30_C120_0']
    names = ['I2_N7_T100_C100_0', 'I2_N7_T100_C120_0', 'I2_N7_T100_C120_0']
    

print(*names, sep="\n")

if RUN:
    for name in names:
        subprocess.run(['python', 'main.py', name])

# aggregate
df = pd.DataFrame(columns=['name', 'objVal', 'VSSobjVal', 'Delta', 'runTime', 'gap', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 
                           'Z1VSS', 'Z2VSS', 'Z3VSS', 'Z4VSS', 'Z5VSS'])

for name in names:
    nameres = name + "_res.xlsx"
    nameresfix = name + "_res_fix.xlsx"
    path ='results/'
    dfres = pd.read_excel(path + nameres, sheet_name='general', index_col='param')
    dfresfix = pd.read_excel(path + nameresfix, sheet_name='general', index_col='param')

    nInst = dfres.loc['Inst name']['value']
    objValue = dfres.loc['objValue']['value']
    runTime = dfres.loc['runtime']['value']
    gap = dfres.loc['gap']['value']
    Z1 = dfres.loc['Z1']['value']
    Z2 = dfres.loc['Z2']['value']
    Z3 = dfres.loc['Z3']['value']
    Z4 = dfres.loc['Z4']['value']
    Z5 = dfres.loc['Z5']['value']
    objValueF = dfresfix.loc['objValue']['value']
    Z1F = dfresfix.loc['Z1']['value']
    Z2F = dfresfix.loc['Z2']['value']
    Z3F = dfresfix.loc['Z3']['value']
    Z4F = dfresfix.loc['Z4']['value']
    Z5F = dfresfix.loc['Z5']['value']
    DELTA = (objValueF - objValue)*100/objValue
    df.loc[len(df)] = [nInst, objValue, objValueF, DELTA, runTime, gap, Z1, Z2, Z3, Z4, Z5, 
                           Z1F, Z2F, Z3F, Z4F, Z5F]
df.to_excel('table_results_newN7T100.xlsx')
