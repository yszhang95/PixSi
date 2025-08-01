import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from scipy.optimize import shgo
import math
from .util import uniform_charge_cum_current_part as current_part

    
def objective_function(params,measurements,sample_param_map):
    chi2=0
    
    for m in measurements:
        if m[2]==5000 or m[2]==0:
            continue
        new_m=0
        #if m[0]==(42,103) and m[1]==1158:
        #    print("min call")
        reg=0
        for p,frac in sample_param_map[(m[0],m[1])]:
            new_m+=(params[p]*frac)
            reg+=params[p]**2
        #if m[0]==(42,103) and m[1]==1158:
        #    print(m[2],new_m)
        chi2+=(m[2]-new_m)**2 #+reg*0.05
        
    fvals.append(chi2)
    return chi2
    
fvals=[]

def solver_2D_scipy_simple(measurements,signals,response):
    options = {
        'maxiter': 10000,  # Increase maximum number of iterations
        'ftol': 1e-9,     # Set a tighter tolerance for convergence
        'xtol': 1e-9,       # Set Tolerance on parameter steps
        'disp': False,      # Enable verbose output to monitor progress
    }
    
    max_readout_time_ticks = 12000
    import matplotlib.pyplot as plt
    initial_guess=[]
    bounds=[]

    for s in signals:
        initial_guess.append(1 if s[5] else 10)
        bounds.append((1,None) if s[5] else (0,None))
    
    
    samples = { (m[0], m[1]): m[2] for m in measurements }

    # Store measurement times per pixel
    measurement_times = {}
    for pixel, time in samples:
        measurement_times.setdefault(pixel,[]).append(time)

    # Sort times for efficient searching
    for pixel in measurement_times:
        measurement_times[pixel].sort()
    
    sorted_signals = sorted(signals, key=lambda x: x[3])
    sample_param_map = {}
    
    R = len(response)-1 # FR radius
    
    from bisect import bisect_left, bisect_right
    
    # Separate stop times and remaining fractions
    signal_stop = {}  # Stop times for (s_id,dy,dz)

    signal_remaining = {}  # Remaining fraction for (s_id,dy,dz)
    
    def process_contributions(s_id, dy, dz, s_pixel, start_time, end_time, conv_q):
        neighbor = ( s_pixel[0] + dy , s_pixel[1]+ dz )
        #print("considered neighbor: ",neighbor)
        if neighbor not in measurement_times:
            return
        #print("found it in meas")
        remaining_charge = conv_q.copy()
        signal_remaining[(s_id, dy, dz)] = remaining_charge
        signal_stop[(s_id, dy, dz)] = None

        times = measurement_times[neighbor]
        start_idx = bisect_left(times, start_time)
        #print("times from meas: ",times)
        #print("found start_idx: ",start_idx)
        for i in range(start_idx, len(times)):
            meas_time = times[i]
            if meas_time > max_readout_time_ticks:
                break
            stop_time = signal_stop[(s_id, dy, dz)]
            if stop_time is not None and meas_time > stop_time:
                break
            idx = int((meas_time - start_time) / (end_time - start_time) * len(conv_q))
            idx = min(max(idx,0.0), len(conv_q) - 1)
            Q_contrib = remaining_charge[idx]
            if abs(Q_contrib) > 0:
                if (neighbor, meas_time) not in sample_param_map:
                    sample_param_map[(neighbor, meas_time)] = []
                sample_param_map[(neighbor, meas_time)].append((s_id, Q_contrib))

                # Only reduce charge if real measurement was made
                if samples[(neighbor, meas_time)] not in [0, 5000]:
                    for j in range(idx, len(remaining_charge)):
                        remaining_charge[j] -= Q_contrib
                        #remaining_charge[j]=max(remaining_charge[j],0.0)

                if all(abs(q) <= 1e-6 for q in remaining_charge):
                    signal_stop[(s_id, dy, dz)] = meas_time
                    break
    for s_id, s_pixel, s_value, s_t_start, s_dt, _ in sorted_signals:
        #if s_pixel!=(40, 107):
        #    continue
        #print("Pixel and range : ",s_pixel,(-R,R+1))
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                #if dy!=0 and dz!=0: continue
                
                kernel = response[abs(dy)][abs(dz)]
                conv_q = current_part(s_value, s_dt, kernel)
                start_time = max(0, s_t_start - len(kernel))
                end_time = s_t_start + s_dt
                print(s_t_start,s_dt)
                conv_q = np.cumsum(conv_q[len(conv_q)-(end_time-start_time):])
                conv_q[conv_q<0]=0
                #if s_id==105 and dy==1 and dz==0:
                #    print(dy,dz,len(conv_q),len(kernel))
                #    print(start_time,end_time,s_t_start,s_dt)
                #    plt.plot(conv_q)
                #    plt.show()
                process_contributions(s_id, dy, dz, s_pixel, start_time, end_time, conv_q)

    #print("Measurements: ",measurements[:5])
    #print("signals: ",signals[:5])
    #print("map : ",sample_param_map)
    #for s in sample_param_map:
    #    print(s,":",[i for i in sample_param_map[s] if i[0]==0])
    #for s in sample_param_map:
    #    print(s,":",[i for i in sample_param_map[s] if i[0]==1])
    #for s in sample_param_map:
    #    print(s,":",[i for i in sample_param_map[s] if i[0]==2])
    #for s in sample_param_map:
    #    print(s,":",[i for i in sample_param_map[s] if i[0]==3])
    #initial_guess=[2.81454701e+04, 6.18979215e-10 ,9.30400812e+03, 6.18979215e-10,7.06547541e-10]
    #initial_guess=[27600, 0 ,28242, 0,0]
    #initial_guess=    [ 2.58286396e+04 ,5.52051509e+03  ,2.09004772e+04 ,6.18979215e-10 ,6.18979215e-10]
    #fcn=objective_function(initial_guess,measurements,sample_param_map)
    result = minimize(objective_function,x0=initial_guess,args=(measurements,sample_param_map),method="L-BFGS-B",options=options,bounds=bounds) #'L-BFGS-B' , SLSQP
        #non gradient optimizers : 'Nelder-Mead' , 'Powell'
    #print("Result: ",result.x)
    #print("measurement : toy")
    #for m in measurements:
    #    new_m=0
    #    if (m[0],m[1]) not in sample_param_map:
    #        continue
    #    for p,frac in sample_param_map[(m[0],m[1])]:
            #print(params[p])
    #        new_m+=(result.x[p]*frac)
    #    print((m[1],m[2])," : ",new_m)
    
    
    
    #plt.plot(fvals)
    #plt.show()
    #result.x=[ 2.58286396e+04 ,5.52051509e+03  ,2.09004772e+04 ,6.18979215e-10 ,6.18979215e-10]
    #result.x,initial_guess
    return [(s[0],s[1],q,s[3],s[4]) for s,q in zip(signals,result.x)],sample_param_map
    


