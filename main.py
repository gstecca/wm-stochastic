from myutils import *
import gurobipy as gb
from gurobipy import GRB
import sys



def build_model(inst : Instance):
    mym = mymodel()
    mm = gb.Model('FSG')
    if LOWER_BOUND:
        #inst.p = 0
        #inst.delta ={k:0 for k in inst.delta}
        pass
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

    # if LOWER_BOUND:
    #    for i in inst.Vp:
    #     for j in inst.Vs:
    #         for s in inst.S:
    #             if j!=i:
    #                 y[i,j,s].ub = 0


    if FIX_SOLUTION:
        dfxres = pd.read_excel('results/' + inst.name + "_mean_res.xlsx", sheet_name='x')
        for index, row in dfxres.iterrows():
            if row['x'] < 0.99:
                x[row['i'], row['j']].ub = 0
            else:
                x[row['i'], row['j']].lb = 1
    if INIT_SOL:
        dfxres = pd.read_excel('results/' + inst.name + "_res.xlsx", sheet_name='x')
        for index, row in dfxres.iterrows():
            if row['x'] < 0.99:
                x[row['i'], row['j']].Start = 0
            else:
                x[row['i'], row['j']].Start = 1
    

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

    if VALID_INEQUALITIES:
        mm.addConstr(  gb.quicksum(x[i,j] for i in inst.Vp for j in inst.Vs  if i != j  ) <= len(inst.V1) + inst.m, name="vi1.1" )
        mm.addConstr(  gb.quicksum(x[i,j] for i in inst.Vp for j in inst.Vs  if i != j ) >= len(inst.V1) , name="vi1.2" )
        mm.addConstrs( ( gb.quicksum( y[i,j,s] for i in inst.Vp for j in inst.Vs if j!=i ) -  gb.quicksum(rho[j,s] for j in range(1, inst.n + 1)  ) <= 0 for s in inst.S), name="vi2" )
    
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
    print('Optimization ended with status (2=optimal, 3=infeasible, 5=inf_or_unbounded, 9=timlimit, 11=interrupted)', mm.Status)


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