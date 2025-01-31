import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from .util import uniform_charge_cum_current as current

def objective_function(params,measurements,kernel_mid,kernel_ind, pixel_block_param_map):
    #measurements come in a form of [ [[block of measurements (time,value)], [],..] , [] ] each array of blocks correspond to a pixel with id == to index in array
    
    kl_mid = len(kernel_mid)
    kl_ind = len(kernel_ind)
    
    #THIS GONNA BE VERY SLOW!!!
    #Calculate charge on all pixles from all params
    idx_param=0
    toy_pixels=[np.zeros(1600) for _ in range(len(measurements)+2)]
    for npi,p in enumerate(measurements):
        if len(p)==0:
            continue
        for nb,b in enumerate(p):
            t_st = params[idx_param]
            dt = b[0][0]+kl_mid-t_st
            Q = params[idx_param+1:idx_param+pixel_block_param_map[npi][nb]]
            idx_param+=pixel_block_param_map[npi][nb]
            for nq,q in enumerate(Q):
                cur_ind=current(q,t_st,dt,kernel_ind)
                cur_mid=current(q,t_st,dt,kernel_mid)
                toy_pixels[npi-1]+=cur_ind
                toy_pixels[npi+1]+=cur_ind
                toy_pixels[npi]+=cur_mid
                t_st+=dt
                dt=16 if nq==0 else 28
    #now current for each pixle needs tp be modified -> basically each measurement in block (except first and last one since they are threshold and zero) reduce accumulated current after it by it's value
    # We can also build chi^2 at the same time
    chi2=0
    for npi,p in enumerate(measurements):
        if len(p)==0:
            chi2+=(toy_pixels[npi][-1]-2500)**2
            continue
        for nb,b in enumerate(p):
            for nm,m in enumerate(b):
                if nm==0 or nm==len(b)-1:
                    continue
                toy_pixels[npi][m[0]+1:]=np.maximum(toy_pixels[npi][m[0]+1:]-m[1],0)
            for nm,m in enumerate(b):
                chi2+=(toy_pixels[npi][m[0]]-m[1])**2
    return chi2+abs(np.sum(params))

def solver_2D_scipy(blocks,kernel_mid,kernel_ind):
    options = {
        'maxiter': 1000,  # Increase maximum number of iterations
        'ftol': 1e-9,     # Set a tighter tolerance for convergence
        'disp': True,      # Enable verbose output to monitor progress
    }
    initial_guess=[]
    pixel_block_param_map=[]
    bounds=[]
    for np,p in enumerate(blocks): # per pixel
        if len(p)==0:
            pixel_block_param_map.append([])
            continue
        block_param_map=[0 for _ in range(len(p))]
        for nb,b in enumerate(p): # per block
            initial_guess_perblock = [b[0][0]]+[0 for _ in range(len(b)-1)]
            initial_guess_perblock[1]=1
            bounds_perblock = [(0, None) for _ in initial_guess_perblock]
            bounds_perblock[0]=(b[0][0]-28,b[0][0]+len(kernel_mid))
            #bounds[0]=(block[0][0]-1,block[0][0]+1)
            bounds_perblock[1]=(1,None) # at least something has to be absorbed at or after the trigger even if charge was accumulated before
            block_param_map[nb]=len(initial_guess_perblock)
            for ig in initial_guess_perblock:
                initial_guess.append(ig)
            for bnd in bounds_perblock:
                bounds.append(bnd)
        pixel_block_param_map.append(block_param_map)
    print("InitialGuess/Bounds Preparation is Done")
    
    
    result = minimize(objective_function,x0=initial_guess,args=(blocks, kernel_mid,kernel_ind,pixel_block_param_map),method="Powell",options=options,bounds=bounds) #'L-BFGS-B' , SLSQP
        #non gradient optimizers : 'Nelder-Mead' , 'Powell'
    #result.x,
    return result.x,pixel_block_param_map
    
