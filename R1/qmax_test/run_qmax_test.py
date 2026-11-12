"""R1.3 check: solve each instance with the original constants (M = Qmax = 1e4)
and with the tight instance-specific constants
    M    = Qbar = max_s sum_j (d_j + sum_t delta_jts)
    Qmax = max(C, Qbar - C)
and compare objective, gap, runtime.

Usage: python run_qmax_test.py <instance_name> <SM|EM> <threads> <out_csv>
Run from a directory containing main.py, myutils.py, params.json and instances/.
"""
import sys, os, json, time
import main as wm
from myutils import load_instance

name, model, threads, out_csv = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

with open('params.json') as fp:
    pms = json.load(fp)
pms.update({'POLICY': 'P0', 'XFIX': {}, 'FIX_SOLUTION': False, 'INIT_SOL': False,
            'LOWER_BOUND': False, 'WRITE_LP': False, 'max_runtime': 3600})
if model == 'EM':
    # base file with delta replaced by delta_expected and LS = 1
    # (same data as the *_mean.xlsx file)
    pms['inst_name'] = name
    pms['INSTANCE_MEAN'] = True
else:
    pms['inst_name'] = name
    pms['INSTANCE_MEAN'] = False

inst = load_instance(pms)
Qbar = max(sum(inst.d[j] + sum(inst.delta[(j, t, s)] for t in inst.T)
               for j in range(1, inst.n + 1)) for s in inst.S)
tight = {'M': float(Qbar), 'Qmax': float(max(inst.C, Qbar - inst.C))}
orig = {'M': float(inst.M), 'Qmax': float(inst.Qmax)}

rows = []
for variant, consts in (('orig', orig), ('tight', tight)):
    inst.M, inst.Qmax = consts['M'], consts['Qmax']
    mym = wm.build_model(inst)
    mm = mym.m
    mm.Params.Threads = threads
    mm.Params.OutputFlag = 0
    mm.Params.LogFile = f'logs/{name}_{model}_{variant}.log'
    wm.run_model(inst, mym)
    ok = mm.SolCount > 0
    rows.append(dict(
        name=name, model=model, n=inst.n, C=inst.C, Qbar=Qbar, variant=variant,
        M=consts['M'], Qmax=consts['Qmax'], status=mm.Status,
        obj=mm.ObjVal if ok else '', bound=mm.ObjBound if ok else '',
        gap=mm.MIPGap if ok else '', runtime=mm.Runtime, nodes=mm.NodeCount,
        maxviol=mm.getAttr('MaxVio') if ok else '',
        x=';'.join(f'{i}-{j}' for (i, j), v in mym.x.items() if v.X > 0.5) if ok else ''))

hdr = not os.path.isfile(out_csv)
with open(out_csv, 'a') as f:
    keys = list(rows[0].keys())
    if hdr:
        f.write(','.join(keys) + '\n')
    for r in rows:
        f.write(','.join(str(r[k]) for k in keys) + '\n')
print(name, model, [(r['variant'], r['obj'], r['runtime']) for r in rows])
