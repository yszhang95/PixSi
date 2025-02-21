import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from scipy.optimize import shgo
import math
from .util import uniform_charge_cum_current as current

def objective_function(params,measurements,kernel_mid,kernel_ind, pixel_block_param_map):
    #measurements come in a form of [ [[block of measurements (time,value)], [],..] , [] ] each array of blocks correspond to a pixel with id == to index in array
    
    kl_mid = len(kernel_mid)
    kl_ind = len(kernel_ind)
    
    #THIS GONNA BE VERY SLOW!!!
    #Calculate charge on all pixles from all params
    idx_param=0
    toy_pixels=np.zeros((len(measurements)+2,1600))
    for npi,p in enumerate(measurements): # per pixel
        if len(p)==0:
            continue
        for nb,b in enumerate(p): # per block
            t_st = params[idx_param]
            if nb>0:
                t_st=max(t_st,p[nb-1][-2][0]+16) if len(p[nb-1])==3 else max(t_st,p[nb-1][-2][0]+28)
            dt = b[0][0]+kl_mid-t_st
            Q = params[idx_param+1:idx_param+pixel_block_param_map[npi][nb]]
            idx_param+=pixel_block_param_map[npi][nb]
            for nq,q in enumerate(Q): # per charge parameter
                cur_ind=current(q,t_st,dt,kernel_ind)
                cur_mid=current(q,t_st,dt,kernel_mid)
                toy_pixels[npi]+=cur_ind
                toy_pixels[npi+2]+=cur_ind
                toy_pixels[npi+1]+=cur_mid
                t_st+=dt
                dt=16 if nq==0 else 28
    #now current for each pixle needs tp be modified -> basically each measurement in block (except first and last one since they are threshold and zero) reduce accumulated current after it by it's value
    #import matplotlib.pyplot as plt
    #plt.plot(toy_pixels[-2])
    #plt.plot(toy_pixels[1])
    #plt.plot(toy_pixels[2])
    #print(measurements)
    #M=np.array(measurements[0][0]+measurements[0][1])
    #plt.scatter(M[:,0],M[:,1])
    # We can also build chi^2 at the same time
    chi2=0
    lastele=0
    for npi,p in enumerate(measurements): # per pixel
        if len(p)==0:
            continue
        for nb,b in enumerate(p): # per block
            for nm,m in enumerate(b): # per measurement
                chi2+=((toy_pixels[npi+1,m[0]]-m[1]))**2
                if nm==0 or nm==len(b)-1:
                    continue
                toy_pixels[npi+1,m[0]+2:]-=toy_pixels[npi+1,m[0]+1] #Thoughts: as trigger works we take time of the measurement and subtrackt next value since 1 ticktime of dead pixle also "lost"
                #toy_pixels[npi+1][m[0]+1:]=toy_pixels[npi+1][m[0]+1:]-m[1]#np.maximum(toy_pixels[npi+1][m[0]+1:]-m[1],0)#another approach is to subtract actual measurement at this point as we know what we measured, but what to do with negative values?
            #for nm,m in enumerate(b): # per measurement
            #    chi2+=((toy_pixels[npi+1,m[0]]-m[1]))**2
    
    #chi2+=(toy_pixels[0][-1])**2
    #chi2+=(toy_pixels[-1][-1])**2
    #plt.plot(toy_pixels[-2])
    #plt.plot(toy_pixels[1])
    #plt.plot(toy_pixels[2])
    #plt.show()
    #print("Function Result: ",chi2+abs(np.sum(params)))
    fvals.append(chi2)#+abs(np.sum(params)))
    return chi2

fvals=[]

def solver_2D_scipy(blocks,kernel_mid,kernel_ind):
    options = {
        'maxiter': 10000,  # Increase maximum number of iterations
        'ftol': 1e-9,     # Set a tighter tolerance for convergence
        'xtol': 1e-9,       # Set Tolerance on parameter steps
        'disp': True,      # Enable verbose output to monitor progress
    }
    initial_guess=[]
    pixel_block_param_map=[]
    bounds=[]
    for npi,p in enumerate(blocks): # per measurement
        if len(p)==0:
            pixel_block_param_map.append([])
            continue
        block_param_map=[0 for _ in range(len(p))]
        for nb,b in enumerate(p): # per block
            initial_guess_perblock = [b[0][0]]+[0 for _ in range(len(b)-1)]
            initial_guess_perblock[1]=1
            bounds_perblock = [(0, None) for _ in initial_guess_perblock]
            #bounds_perblock[0]=(b[0][0]-16,b[0][0]+len(kernel_mid))
            bounds_perblock[0]=(b[0][0],b[0][0])
            #bounds[0]=(block[0][0]-1,block[0][0]+1)
            bounds_perblock[1]=(1,None) # at least something has to be absorbed at or after the trigger even if charge was accumulated before
            block_param_map[nb]=len(initial_guess_perblock)
            for ig in initial_guess_perblock:
                initial_guess.append(ig)
            for bnd in bounds_perblock:
                bounds.append(bnd)
        pixel_block_param_map.append(block_param_map)
    print("InitialGuess/Bounds Preparation is Done")
    print("Initial Guess: ",initial_guess)
    print("Measurements: ",blocks)
    print("bounds: ",bounds)
    #initial_guess=    [245, 2.58286396e+04 ,5.52051509e+03 ,319, 2.09004772e+04 ,6.18979215e-10 ,6.18979215e-10]
    #initial_guess=[1.48623511e+02, 5.25540329e+04, 1.07054458e-08, 6.18979215e-10,1.73294224e+02, 5.35757061e+04, 3.15151257e-09, 6.18979215e-10,1.86485364e+02, 5.51951181e+04, 6.18979215e-10, 6.18979215e-10,1.95495223e+02, 4.07848474e+04, 6.18979219e-10, 6.18979215e-10,1.74506143e+02, 1.06787135e+04, 6.19101253e-10,]
    #initial_guess=[245,2.92590540e+04, 6.18979215e-10,319 ,7.16497931e+03, 6.18978839e-10, 5.08666863e+05]
    #f=objective_function(initial_guess,blocks, kernel_mid,kernel_ind,pixel_block_param_map)
    result = minimize(objective_function,x0=initial_guess,args=(blocks, kernel_mid,kernel_ind,pixel_block_param_map),method="Powell",options=options,bounds=bounds) #'L-BFGS-B' , SLSQP
        #non gradient optimizers : 'Nelder-Mead' , 'Powell'
    print("Result: ",result.x)
    import matplotlib.pyplot as plt
    
    plt.plot(fvals)
    plt.show()
    #result.x,initial_guess
    return result.x,pixel_block_param_map
    
