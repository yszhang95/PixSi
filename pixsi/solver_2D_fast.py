import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from scipy.optimize import shgo
import math
from .util import uniform_charge_cum_current as current
from .util import uniform_charge_cum_current_part as current_part

    
def objective_function(params,measurements,sample_param_map):
    chi2=0
    
    #sample_param_map = smooth_charge_distribution(measurements, sample_param_map)


    #print(params)
    meas=[]
    toys=[]
    for m in measurements:
        new_m=0
        #print(sample_param_map[(m[0],m[1])])
        for p,frac in sample_param_map[(m[0],m[1])]:
            #print(params[p])
            new_m+=(params[p]*frac)
            #if params[p]>0: coeff*=params[p]
        meas.append((m[1],m[2]))
        toys.append((m[1],new_m))
        #print("comparison:",m,s,coeff)
        chi2+=(m[2]-new_m)**2
        
        
    
    #import matplotlib.pyplot as plt
    #m_bb=np.array(meas)
    #t_bb=np.array(toys)
    #plt.scatter(m_bb[:,0],m_bb[:,1],label="meas")
    #plt.scatter(t_bb[:,0],t_bb[:,1],label="toy")
    #plt.legend()
    #plt.show()
    #print("chi^2=",chi2)
    fvals.append(chi2)
    return chi2
    
fvals=[]

def solver_2D_scipy(measurements,signals,response):
    options = {
        'maxiter': 10000,  # Increase maximum number of iterations
        'ftol': 1e-9,     # Set a tighter tolerance for convergence
        'xtol': 1e-9,       # Set Tolerance on parameter steps
        'disp': True,      # Enable verbose output to monitor progress
    }

    initial_guess=[]
    bounds=[]

    for s in signals:
        if s[5]:
            initial_guess.append(1)
            bounds.append((1,None))
        else:
            initial_guess.append(0)
            bounds.append((0,None))
    
    samples = { (m[0], m[1]): m[2] for m in measurements }

    # Store measurement times per pixel
    measurement_times = {}
    for (pixel, time) in samples:
        if pixel not in measurement_times:
            measurement_times[pixel] = []
        measurement_times[pixel].append(time)

    # Sort times for efficient searching
    for pixel in measurement_times:
        measurement_times[pixel].sort()
    
    sorted_signals = sorted(signals, key=lambda x: x[3])
    sample_param_map = {}
    
    L_self = len(response[0])  # Kernel length for self-response
    L_neighbor = len(response[1])  # Kernel length for neighbors
    from bisect import bisect_left, bisect_right
    
    # Separate stop times and remaining fractions
    signal_stop_self = {}  # Stop times for self-responses
    signal_stop_neighbor = {}  # Stop times for each neighbor {s_id: {neighbor: stop_time}}

    signal_remaining_self = {}  # Remaining fraction for self-response
    signal_remaining_neighbor = {}  # Remaining fraction for each neighbor {s_id: {neighbor: fraction}}


    def compute_contribution(meas_time, start, end, conv_q):
        if meas_time < start:
            return 0
        elif meas_time >= end:
            return conv_q[-1]
        else:
            idx = int((meas_time - start) / (end - start) * len(conv_q))
        return conv_q[min(idx, len(conv_q) - 1)]
    
    
    for s_id, s_pixel, s_value, s_t_start, s_dt,_ in sorted_signals:
        
        conv_q = [current_part(s_value,s_dt,response[0]),current_part(s_value,s_dt,response[1])]
        #print(s_id," ",len(conv_q[0]),s_dt,s_t_start)
        # Compute affected time ranges
        start_time_self = max(0, s_t_start - L_self)
        end_time_self = s_t_start + s_dt
        start_time_neighbor = max(0, s_t_start - L_neighbor)
        end_time_neighbor = end_time_self
        
        signal_stop_self[s_id] = None
        signal_remaining_self[s_id] = conv_q[0].copy()  # Keep track of remaining charge

        signal_stop_neighbor[s_id] = {}
        signal_remaining_neighbor[s_id] = {}

        # Neighboring pixels
        neighbors = [s_pixel - 1, s_pixel + 1]
        for neighbor in neighbors:
            signal_stop_neighbor[s_id][neighbor] = None
            signal_remaining_neighbor[s_id][neighbor] = conv_q[1].copy()

        def process_contributions(pixel, start_time, end_time, is_self):
            if pixel not in measurement_times:
                return
        
            times = measurement_times[pixel]
            start_idx = bisect_left(times, start_time)
        
            for i in range(start_idx, len(times)):
                meas_time = times[i]
                if meas_time > 1600:
                    break
                
                if is_self:
                    stop_time = signal_stop_self[s_id]
                    remaining_charge = signal_remaining_self[s_id]
                else:
                    stop_time = signal_stop_neighbor[s_id][pixel]
                    remaining_charge = signal_remaining_neighbor[s_id][pixel]
                
                if stop_time is not None and meas_time > stop_time:
                    break
                
                length = len(conv_q[0]) if is_self else len(conv_q[1])
                idx = int((meas_time - start_time) / (end_time - start_time) * length)
                #print(meas_time,start_time,idx)
                idx = min(idx, length - 1)  # Ensure index is within bounds
                #print("idx:",idx,remaining_charge[idx])
            # Get the charge contribution from conv_q
                Q_contrib = remaining_charge[idx]
            
                if Q_contrib > 0:
                    if (pixel, meas_time) not in sample_param_map:
                        sample_param_map[(pixel, meas_time)] = []
                    sample_param_map[(pixel, meas_time)].append((s_id, Q_contrib))
                
  

                    # Only reduce future contributions if the measurement value is NOT 5000
                    if samples[(pixel, meas_time)] != 5000 and samples[(pixel, meas_time)] != 0:
                        for j in range(idx, len(remaining_charge)):
                            remaining_charge[j] -= Q_contrib
                
                    # Stop future contributions if effectively depleted
                    if all(q <= 1e-6 for q in remaining_charge):  # Avoid floating-point issues
                        if is_self:
                            signal_stop_self[s_id] = meas_time
                        else:
                            signal_stop_neighbor[s_id][pixel] = meas_time
                        break

    # Process self and neighbor responses
        process_contributions(s_pixel, start_time_self, end_time_self, is_self=True)
    
    # Process each neighbor separately
        for neighbor in neighbors:
            process_contributions(neighbor, start_time_neighbor, end_time_neighbor, is_self=False)
    

    #print("Measurements: ",measurements)
    #print("samples: ",samples)
    #print("signals: ",signals )
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
    result = minimize(objective_function,x0=initial_guess,args=(measurements,sample_param_map),method="Powell",options=options,bounds=bounds) #'L-BFGS-B' , SLSQP
        #non gradient optimizers : 'Nelder-Mead' , 'Powell'
    print("Result: ",result.x)
    print("measurement : toy")
    for m in measurements:
        new_m=0
        for p,frac in sample_param_map[(m[0],m[1])]:
            #print(params[p])
            new_m+=(result.x[p]*frac)
        print((m[1],m[2])," : ",new_m)
    
    
    import matplotlib.pyplot as plt
    
    plt.plot(fvals)
    plt.show()
    #result.x=[ 2.58286396e+04 ,5.52051509e+03  ,2.09004772e+04 ,6.18979215e-10 ,6.18979215e-10]
    #result.x,initial_guess
    return [(s[0],s[1],s[2]*q,s[3],s[4]) for s,q in zip(signals,result.x)],sample_param_map
    


