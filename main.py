from myutils import *
import gurobipy as gb
from gurobipy import GRB
import sys

class mymodel:
    def __init__(self):
        self.m : gb.Model = None
        self.x = {}
        self.U = {}
        self.TBar = {}
        self.y = {}
        self.Q = {}
        self.R = {}
        self.L = {}
        self.rho = {}
        self.lambd = {}
        self.alpha = {}
        self.Z1 = None
        self.Z2 = None
        self.Z3 = None
        self.Z4 = None
    def set(self, m, x, U, TBar, y, Q, R, L, rho, lambd, alpha, Z1, Z2, Z3, Z4, Z5):
        self.m = m
        self.x = x
        self.U = U
        self.TBar =TBar
        self.y = y
        self.Q = Q
        self.R = R
        self.L = L
        self.rho = rho
        self.lambd = lambd
        self.alpha = alpha
        self.Z1 = Z1
        self.Z2 = Z2
        self.Z3 = Z3
        self.Z4 = Z4
        self.Z5 = Z5

    def to_excel(self, inst: Instance):
        filename = 'results/' + inst.name + "_res" 
        filename += '_fix.xlsx' if FIX_SOLUTION==True  else '.xlsx'
        # create a excel writer object
        df_gen = pd.DataFrame(columns=['param', 'value'])
        df_gen.loc[len(df_gen)] = ['Inst name', inst.name]
        df_gen.loc[len(df_gen)] = ['objValue', self.m.ObjVal]
        df_gen.loc[len(df_gen)] = ['runtime', self.m.Runtime]
        df_gen.loc[len(df_gen)] = ['gap', self.m.MIPGap]
        df_gen.loc[len(df_gen)] = ['Z1', self.Z1.getValue()]
        df_gen.loc[len(df_gen)] = ['Z2', self.Z2.getValue()]
        df_gen.loc[len(df_gen)] = ['Z3', self.Z3.getValue()]
        df_gen.loc[len(df_gen)] = ['Z4', self.Z4.getValue()]
        df_gen.loc[len(df_gen)] = ['Z5', self.Z5.getValue()]

        df_x = pd.DataFrame()
        df_x['i'] = [k[0] for k in self.x.keys() ]
        df_x['j'] = [k[1] for k in self.x.keys() ]
        df_x['x'] = [v.X for v in self.x.values() ]
        df_x = df_x[df_x['x'] > 1e-4]

        df_U = pd.DataFrame()
        df_U['j'] = [k[0] for k in self.U.keys() ]
        df_U['t'] = [k[1] for k in self.U.keys() ]
        df_U['U'] = [v.X for v in self.U.values() ]
        df_U = df_U[df_U['U'] > 1e-4]

        df_TBar = pd.DataFrame()
        df_TBar['i'] = [k for k in self.TBar.keys() ]
        df_TBar['TBar'] = [v.X for v in self.TBar.values() ]

        df_y = pd.DataFrame()
        df_y['i'] = [k[0] for k in self.y.keys() ]
        df_y['j'] = [k[1] for k in self.y.keys() ]
        df_y['s'] = [k[2] for k in self.y.keys() ]
        df_y['y'] = [v.X for v in self.y.values() ]
        df_y = df_y[df_y['y'] > 1e-4]

        df_Q = pd.DataFrame()
        df_Q['j'] = [k[0] for k in self.Q.keys() ]
        df_Q['s'] = [k[1] for k in self.Q.keys() ]
        df_Q['Q'] = [v.X for v in self.Q.values() ]

        df_R = pd.DataFrame()
        df_R['j'] = [k[0] for k in self.R.keys() ]
        df_R['s'] = [k[1] for k in self.R.keys() ]
        df_R['R'] = [v.X for v in self.R.values() ]
 
        df_L = pd.DataFrame()
        df_L['j'] = [k[0] for k in self.L.keys() ]
        df_L['s'] = [k[1] for k in self.L.keys() ]
        df_L['L'] = [v.X for v in self.L.values() ]

        df_rho = pd.DataFrame()
        df_rho['j'] = [k[0] for k in self.rho.keys() ]
        df_rho['s'] = [k[1] for k in self.rho.keys() ]
        df_rho['rho'] = [v.X for v in self.rho.values() ]
        df_rho = df_rho[df_rho['rho'] > 1e-4]

        #df_lambd = pd.DataFrame()
        #df_lambd['j'] = [k[0] for k in self.lambd.keys() ]
        #df_lambd['s'] = [k[1] for k in self.lambd.keys() ]
        #df_lambd['lambda'] = [v.X for v in self.lambd.values() ]

        df_alpha = pd.DataFrame()
        df_alpha['j'] = [k[0] for k in self.alpha.keys() ]
        df_alpha['s'] = [k[1] for k in self.alpha.keys() ]
        df_alpha['alpha'] = [v.X for v in self.alpha.values() ]
        df_alpha = df_alpha[df_alpha['alpha'] > 1e-4]

        with pd.ExcelWriter(filename) as writer:
            df_gen.to_excel(writer, sheet_name="general", index=False)
            df_x.to_excel(writer, sheet_name="x", index=False)
            df_U.to_excel(writer, sheet_name="U", index=False)
            df_TBar.to_excel(writer, sheet_name="TBar", index=False)
            df_y.to_excel(writer, sheet_name="y", index=False)
            df_Q.to_excel(writer, sheet_name="Q", index=False)
            df_R.to_excel(writer, sheet_name="R", index=False)
            df_L.to_excel(writer, sheet_name="L", index=False)
            df_rho.to_excel(writer, sheet_name="rho", index=False)
            #df_lambd.to_excel(writer, sheet_name="lambda", index=False)
            df_alpha.to_excel(writer, sheet_name="alpha", index=False)     


def build_model(inst : Instance):
    mym = mymodel()
    mm = gb.Model('FSG')
    x = {(i,j) : mm.addVar(vtype = GRB.BINARY, name='x_{}_{}'.format(i,j)) for i in inst.Vp for j in inst.Vs if i != j }
    U = {(j,t) : mm.addVar(vtype = GRB.BINARY, name='U_{}_{}'.format(j,t)) for j in inst.Vs for t in inst.T}
    TBar = {i : mm.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = 'TBar_{}'.format(i)) for i in inst.V}
    y = {(i,j,s) :mm.addVar(vtype = GRB.BINARY, name = 'y_{}_{}_{}'.format(i,j,s)) for i in inst.Vp for j in inst.Vs for s in inst.S if j!=i }
    Q = {(j,s) : mm.addVar(vtype=  GRB.CONTINUOUS, lb = 0, name = 'Q_{}_{}'.format(j,s)) for j in inst.V for s in inst.S}
    R = {(j,s) : mm.addVar(vtype=  GRB.CONTINUOUS, lb = 0, name = 'R_{}_{}'.format(j,s)) for j in range(inst.n+1, inst.n+inst.m+1) for s in inst.S}
    L = {(j,s) : mm.addVar(vtype=  GRB.CONTINUOUS, lb = 0, name = 'L_{}_{}'.format(j,s)) for j in range(1, inst.n + 1) for s in inst.S}
    rho = {(j,s) : mm.addVar(vtype=  GRB.BINARY,  name = 'rho_{}_{}'.format(j,s)) for j in range(1, inst.n + 1) for s in inst.S}
    lambd = {(j,s) : mm.addVar(vtype=  GRB.BINARY,  name = 'lambd_{}_{}'.format(j,s)) for j in range(1, inst.n + 1) for s in inst.S}
    alpha = {(j,s) : mm.addVar(vtype=  GRB.BINARY,  name = 'alpha_{}_{}'.format(j,s)) for j in range(1, inst.n + 1) for s in inst.S}

    mm.update()

    ZOB1 = gb.quicksum(inst.c[i,j] * x[i,j] for i in inst.Vp for j in inst.Vs if j != i) 
    ZOB2 = (1/inst.LS) * gb.quicksum(inst.c[i,j] * y[i,j,s] for s in inst.S for i in range(1, inst.n+1) for j in inst.Vs if j != i)
    ZOB3 = (1/inst.LS) * gb.quicksum(inst.c[j,0] * alpha[j,s] for s in inst.S for j in inst.V1)
    ZOB4 = (1/inst.LS) * gb.quicksum(inst.p * R[j,s] for s in inst.S for j in range(inst.n + 1, inst.n + inst.m + 1) )
    ZOB5 = (1/inst.LS) * gb.quicksum(inst.p * L[j,s] for s in inst.S for j in inst.V1)
    ZOB = ZOB1 - ZOB2 + ZOB3 + ZOB4 + ZOB5

    if FIX_SOLUTION:
        dfxres = pd.read_excel('results/' + inst.name + "_mean_res.xlsx", sheet_name='x')
        for index, row in dfxres.iterrows():
            if row['x'] < 0.99:
                continue
            x[row['i'], row['j']].lb = 1
    

    mm.setObjective(ZOB, GRB.MINIMIZE)
    mm.update()

    mm.addConstrs((gb.quicksum(x[i,j] for j in inst.Vs if j!=i) - gb.quicksum(U[i,t] for t in inst.T) == 0 for i in inst.V1  ), name='ct02.1')
    mm.addConstrs((gb.quicksum(x[j,i] for j in inst.Vp if j!=i) - gb.quicksum(U[i,t] for t in inst.T) == 0 for i in inst.V1  ), name='ct02.2')
    mm.addConstr( gb.quicksum(x[0,j] for j in inst.Vs)  == inst.m , name = 'ct03' )
    mm.addConstrs( (gb.quicksum(x[j,i] for j in inst.Vp) == 1 for i in range (inst.n+1, inst.n + inst.m + 1) ), name='ct04'  )

    mm.addConstrs((gb.quicksum(U[j,t] for t in inst.T) == 1 for j in inst.Vs), name = 'ct05')
    mm.addConstr(TBar[0] == 0, name='ct06')
    mm.addConstrs(( TBar[j] - TBar[i] - inst.t[i,j] + inst.Tmax * (1 - x[i,j]) >=0 for i in inst.Vp for j in inst.Vs if j != i ), name='ct07' )
    mm.addConstrs(( inst.etw[t] - inst.Tmax * (1 - U[j,t]) - TBar[j]  <= 0 for j in inst.Vs for t in inst.T), name = 'ct08.1')
    mm.addConstrs(( TBar[j]  - inst.ltw[t] - inst.Tmax * (1 - U[j,t]) <= 0 for j in inst.Vs for t in inst.T), name = 'ct08.2')

    mm.addConstrs((Q[0,s] == 0  for s in inst.S), name = 'ct09')
    mm.addConstrs((Q[j,s] - Q[i,s] + inst.M * (1 - x[i,j]) - inst.d[j] - gb.quicksum( inst.delta[j,tau,s] * U[j,t] for t in inst.T for tau in range(1, t+1 ))  >= 0
                   for s in inst.S for i in inst.Vp for j in range(1, inst.n + 1) if j != i), name = 'ct10')
    mm.addConstrs((Q[j,s] - Q[i,s] - inst.M * (1 - x[i,j]) - inst.d[j] - gb.quicksum( inst.delta[j,tau,s] * U[j,t] for t in inst.T for tau in range(1, t+1 ))  <= 0
                   for s in inst.S for i in inst.Vp for j in range(1, inst.n + 1) if j != i), name = 'ct10_new')
    mm.addConstrs((Q[j,s] - Q[i,s] + inst.M * (1 - x[i,j]) >= 0 for s in inst.S for i in inst.Vp for j in range(inst.n + 1, inst.n + inst.m + 1)), name = 'ct11')
    mm.addConstrs((Q[j,s] - Q[i,s] - inst.M * (1 - x[i,j]) <= 0 for s in inst.S for i in inst.Vp for j in range(inst.n + 1, inst.n + inst.m + 1)), name = 'ct11_new')
    mm.addConstrs((R[j,s] - Q[j,s] + inst.C >= 0 for s in inst.S for j in range(inst.n + 1, inst.n + inst.m + 1)), name = 'ct12') 
    #mm.addConstrs((R[j,s]  >= 0 for s in inst.S for j in range(inst.n + 1, inst.n + inst.m + 1)), name = 'ct13') already considered in variable definition
    
    mm.addConstrs((L[j,s] - gb.quicksum( inst.delta[j,tau,s] * U[j,t] for t in inst.T for tau in range(t+1, inst.LT + 1)) == 0 for s in inst.S for j in inst.V1), name = 'ct14')

    mm.addConstrs((rho[j,s] - ((1/inst.Qmax)*(Q[j,s] - inst.C)) >= 0 for s in inst.S for j in inst.V1), name = 'ct15')
    mm.addConstrs((rho[j,s] + ((1/inst.Qmax)*(inst.C - Q[j,s] )) <= 1 for s in inst.S for j in inst.V1), name = 'ct16') 

    #mm.addConstrs((rho[j,s] + inst.Qmax*R[j,s] >= 1 for s in inst.S for j in range(1, inst.n + 1)) , name = 'ct_rho_new')

    mm.addConstrs((y[i,j,s] - 0.5 * (x[i,j] + rho[i,s]) <= 0 for s in inst.S for i in inst.V1 for j in inst.Vs if j != i), name = 'ct17')

    mm.addConstrs( (y[0,j,s] == 0 for s in inst.S for j in inst.Vs), name='cty0')
    mm.addConstrs(( alpha[j,s] == gb.quicksum(y[j,i,s] for i in inst.Vs if j!=i) - gb.quicksum(y[i,j,s] for i in range(1, inst.n+1) if j!=i)  for s in inst.S for j in range(1, inst.n + 1)), name = 'ctyalpha')
    
    """
    mm.addConstr(x[0,1]==1)
    mm.addConstr(x[1,2]==1)
    mm.addConstr(x[2,3]==1)
    mm.addConstr(x[3,4]==1)
    mm.addConstr(x[4,5]==1)
    mm.addConstr(x[5,11]==1)
    mm.addConstr(x[0,10]==1)
    mm.addConstr(x[10,9]==1)
    mm.addConstr(x[9,8]==1)
    mm.addConstr(x[8,7]==1)
    mm.addConstr(x[7,6]==1)
    mm.addConstr(x[6,12]==1)
    """
    
    mm.update()
    if (WRITE_LP):
        mm.write('results/model_'+inst.name+'.lp')

    mym.set(mm, x, U, TBar, y, Q, R, L, rho, lambd, alpha, ZOB1, ZOB2, ZOB3, ZOB4, ZOB5)
    return mym


def run_model(inst : Instance, mym : mymodel):
    mm = mym.m
    mm.Params.TimeLimit = max_runtime
    #mm.Params.IntFeasTol = 1e-7
    mm.optimize()
    print('Optimizatoin ended with status (2=optimal, 3=infeasible, 5=inf_or_unbounded, 9=timlimit, 11=interrupted)', mm.Status)


if __name__ == "__main__":
    #inst_name = 'I1_S1_mean'
    inst_name = 'I1_S4'
    if len(sys.argv) > 1:
        inst_name = sys.argv[1]

    print("###### Processing Instance named: ", inst_name, '   #############')
    filename = 'instances/'+inst_name+'.xlsx'
    inst = load_instance(filename)
    print ("loaded instance with ", inst.n, ' nodes' )
    print (inst.to_string())

    print("building model")
    mym = build_model(inst)

    print("run model")
    run_model(inst, mym)

    print("save solution to excel (if a solution has been found)")
    if mym.m.Status in [2,9,11]:
        print('object value: ', mym.m.ObjVal)
        print(mym.Z1.getValue())
        print(mym.Z2.getValue())
        print(mym.Z3.getValue())
        print(mym.Z4.getValue())
        print(mym.Z5.getValue())
        fsolname = 'results/out_'+inst.name
        fsolname += '_fix.sol' if FIX_SOLUTION==True  else '.sol'
        mym.m.write(fsolname)
        mym.to_excel(inst)