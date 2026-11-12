"""Compare orig (M = Qmax = 1e4) vs tight constants. Usage: python analyze.py results_dir"""
import sys, glob
import pandas as pd

d = pd.concat([pd.read_csv(f) for f in glob.glob(sys.argv[1] + '/res_*.csv')])
o = d[d.variant == 'orig'].set_index(['name', 'model'])
t = d[d.variant == 'tight'].set_index(['name', 'model'])
c = o.join(t, lsuffix='_o', rsuffix='_t')
c['dobj'] = (c.obj_t - c.obj_o).abs() / c.obj_o.abs().clip(lower=1)
c['ratio_rt'] = c.runtime_t / c.runtime_o
c['both_opt'] = (c.status_o == 2) & (c.status_t == 2)
c['same_x'] = c.x_o == c.x_t

print('pairs:', len(c), ' status counts orig/tight:',
      o.status.value_counts().to_dict(), t.status.value_counts().to_dict())
print('max |rel obj diff|:', c.dobj.max(), ' pairs with diff > 1e-6:', (c.dobj > 1e-6).sum())
print('max constraint violation orig/tight:', o.maxviol.max(), t.maxviol.max())
print('Qmax tight range:', t.Qmax.min(), t.Qmax.max(), ' M tight range:', t.M.min(), t.M.max())
print('identical routes:', c.same_x.sum(), '/', len(c))
g = c.groupby(['model', 'n_o'])
s = pd.DataFrame({
    'count': g.size(),
    'rt_orig_mean': g.runtime_o.mean(), 'rt_tight_mean': g.runtime_t.mean(),
    'rt_ratio_geomean': g.ratio_rt.apply(lambda r: float((r.clip(lower=1e-9)).prod() ** (1 / len(r)))),
    'nodes_orig_mean': g.nodes_o.mean(), 'nodes_tight_mean': g.nodes_t.mean(),
    'max_rel_obj_diff': g.dobj.max(),
})
print(s.round(4).to_string())
print(c[c.dobj > 1e-6][['obj_o', 'obj_t', 'gap_o', 'gap_t', 'status_o', 'status_t']])
c.to_csv(sys.argv[1] + '/comparison.csv')
