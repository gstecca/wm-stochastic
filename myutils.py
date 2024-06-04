import pandas as pd
import math
from params import *

class node:
    def __init__(self, i, x, y):
        self.i = i
        self.x = x
        self.y = y

def get_distance(i: node, j: node):
    return math.dist([i.x, i.y], [j.x, j.y])


class Instance:
    def __init__(self):
        self.name = 'myname'
        self.n = 1
        self.nodes = {} 
        self.LT = 1
        self.T = [1]
        self.m = 4
        self.LS = 1
        self.S = [1]
        self.C = 10
        self.p = 10
        self.Tmax = 10e6
        self.Qmax = 10e6
        self.M = 10e6
        self.e = {}# (i,j) : distance or cost
        self.c = {} 
        self.t = {}
        self.etw = {} # t:e
        self.ltw = {} # t:l
        self.d = {} # i:d
        self.delta = {} # (j,t,s):delta
        self.V = set()
        self.Vp = set()
        self.Vs = set()
        self.V1 = set()

    def fill_edges(self):
        for i in self.nodes.values():
            for j in self.nodes.values():
                dist = get_distance(i,j)
                self.c[i.i, j.i] = dist * D2C
                self.t[i.i, j.i] = dist * D2T
                self.e[i.i, j.i] = dist 


    def fillV(self):
        self.V = set(i for i in range(self.n+self.m+1))
        self.V1 = set(i for i in range(1, self.n + 1))
        self.Vp = set.difference(self.V, set(i for i in range(self.n+1, self.n+self.m+1)))
        self.Vs =  set.difference(self.V, {0})
        return
    
    def expandNetwork(self):
        # add edges 0->i for i in [n+1,m] with cost and time equal to 0
        ext_e = {}
        ext_c = {}
        ext_t = {}
        for i in range(self.n+1, self.n + self.m + 1):
            ext_e[0,i] = 0
            ext_c[0,i] = 0
            ext_t[0,i] = 0
        # add edge (i,j) for each i in {1,n} j in {n+1, n+m} add e[i,j] con costo c[e[i,j]]= c[i,0]
        for j in range(self.n +1, self.m  + self.n + 1):
            for i in range(1, self.n+1):
                ext_e[i,j] = self.e[i,0] 
                ext_c[i,j] = self.c[i,0] 
                ext_t[i,j] = self.t[i,0]  
        self.e.update(ext_e)
        self.c.update(ext_c)
        self.t.update(ext_t)
        return

    def to_string(self):
        s = ""
        s += 'name: ' + str(self.name ) + '\n'
        s += 'n: ' + str(self.n ) + '\n'
        s += 'LT: ' + str(self.LT) + '\n'
        s += 'T: ' + str(self.T) + '\n'
        s += 'm: ' + str(self.m) + '\n'
        s += 'LS: ' + str(self.LS) + '\n'
        s += 'S: ' + str(self.S) + '\n'
        s += 'C: ' + str(self.C) + '\n'
        s += 'p: ' + str(self.p) + '\n'
        s += 'Tmax: ' + str(self.Tmax) + '\n'
        s += 'Qmax: ' + str(self.Qmax) + '\n'
        s += 'M: ' + str(self.M) + '\n'
        s += 'e: ' + str(self.e) + '\n'
        s += 'etw: ' + str(self.etw) + '\n'
        s += 'ltw: ' + str(self.ltw) + '\n'
        s += 'd: ' + str(self.d) + '\n'
        s += 'delta: ' + str(self.delta)  + '\n'
        s += 'V: ' + str(self.V)  + '\n'
        s += 'Vp: ' + str(self.Vp)  + '\n'
        s += 'Vs: ' + str(self.Vs)  + '\n'
        s += 'V1: ' + str(self.V1)  + '\n'

        return s



def load_instance(filename):
    dfp = pd.read_excel(filename, sheet_name='params', index_col='param')
    dfn = pd.read_excel(filename, sheet_name='nodes')
    dfe = pd.read_excel(filename, sheet_name='edges', dtype={'i':int, 'j':int, 'c':float, 't':float})
    dftw = pd.read_excel(filename, sheet_name='time_windows')
    dfd = pd.read_excel(filename, sheet_name='demand')
    dfdelta = pd.read_excel(filename, sheet_name='delta')
    
    inst = Instance()
    inst.name = dfp.loc['name','value']
    inst.n = dfp.loc['n','value']
    inst.LT = dfp.loc['LT','value']
    inst.T = [t for t in range(1, inst.LT+1)]
    inst.m = dfp.loc['m','value']
    inst.LS = dfp.loc['LS','value']
    inst.S = [s for s in range(1, inst.LS + 1)]
    inst.C = dfp.loc['C','value']
    inst.p = dfp.loc['p','value']
    inst.Tmax = dfp.loc['Tmax','value']
    inst.Qmax = dfp.loc['Qmax','value']
    inst.M = dfp.loc['M','value']

    if LOAD_NODES:
        for index, row in dfn.iterrows():
            inst.nodes[ row["i"] ] = node( row['i'], row["x"], row["y"])
        inst.fill_edges() 
    else:
        for index, row in dfe.iterrows():
            inst.e[ (row["i"], row["j"]) ] = row["c"]
            inst.c[ (row["i"], row["j"]) ] = row["c"]
            inst.t[ (row["i"], row["j"]) ] = row["t"] 

    for index, row in dftw.iterrows():
        inst.etw[ row['t']] =  row['e']
        inst.ltw[ row['t']] =  row['l']
    
    for index, row in dfd.iterrows():
        inst.d[ row['j']] =  row['d']

    for index, row in dfdelta.iterrows():
        inst.delta[ (row['j'], row['t'], row['s']) ] = row['delta']

    inst.fillV()
    inst.expandNetwork()
    if WRITE_EDGES:
        df = pd.DataFrame(columns = ['i', 'j', 'c','t'])
        df['i'] = [k[0] for k, v in inst.e.items()]
        df['j'] = [k[1] for k, v in inst.e.items()]
        df['c'] = [v for k, v in inst.c.items()]
        df['t'] = [v for k, v in inst.t.items()] 
        df.to_excel(filename[0:-4] + '_edges.xlsx', sheet_name='edges', index=None)

    return inst
