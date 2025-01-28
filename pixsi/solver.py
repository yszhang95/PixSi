import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from .util import uniform_charge_cum_current

def objective_function(params, measurements, kernel):
    t_st=params[0]
    q=params[1:]
    func=0
    kernel=kernel[kernel!=0]
    kernel_len=len(kernel)
    shift=measurements[0]
    currs=[]
    fs=[]
    for n,m in enumerate(measurements):
        if n==0:
            c=uniform_charge_cum_current(q[n],t_st,m[0]+kernel_len-t_st,kernel)
            #f=min((m[1]-c[m[0]])**2,(m[1]-c[m[0]-1])**2,(m[1]-c[m[0]+1])**2)
            f=(m[1]-c[m[0]])**2
            currs.append(c)
            fs.append(f)
        elif n==1:
            prevm=measurements[n-1]
            c1=currs[-1].copy()+uniform_charge_cum_current(q[n],prevm[0]+kernel_len,m[0]-prevm[0],kernel)
            currs.append(c1)
            f=(m[1]-c1[m[0]])**2
            fs.append(f)
        elif n>1 and n<len(measurements)-1:
            prevm=measurements[n-1]
            #print(n,q[n])
            c1=currs[-1].copy()+uniform_charge_cum_current(q[n],prevm[0]+kernel_len,m[0]-prevm[0],kernel)
            c1[:prevm[0]]=0
            c1[prevm[0]:]=np.maximum(c1[prevm[0]:]-prevm[1],0)
            currs.append(c1)
            f=(m[1]-c1[m[0]])**2
            fs.append(f)
        elif n==len(measurements)-1:
            prevm=measurements[n-1]
            c1=currs[-1].copy()# we can add zero q past measurement of dt=kernel_len +uniform_charge_cum_current(0,prevm[0]+32,32)
            c1[:prevm[0]]=0
            c1[prevm[0]:]=np.maximum(c1[prevm[0]:]-prevm[1],0)
            currs.append(c1)

            f=(m[1]-c1[m[0]])**2
            fs.append(f)
    #W=np.array([1,1,1,1,1])
    return np.sum(fs)


def solver_scipy(block,kernel):
    options = {
        'maxiter': 10000,  # Increase maximum number of iterations
        'ftol': 1e-9,     # Set a tighter tolerance for convergence
        'disp': True,      # Enable verbose output to monitor progress
        'eps': 0.01       # define Minimizer step
    }
    initial_guess = [block[0][0]]+[0 for _ in range(len(block)-1)]
    bounds = [(0, None) for _ in initial_guess]
    bounds[1]=(1,None) # at least something has to be absorbed at or after the trigger even if charge was accumulated before
    print("Considered Block: ",block)
    print("Initial Guess: ",initial_guess)
    result = minimize(objective_function,x0=initial_guess,args=(block, kernel),method="SLSQP",options=options,bounds=bounds) #'L-BFGS-B' , SLSQP
    return result.x
    
