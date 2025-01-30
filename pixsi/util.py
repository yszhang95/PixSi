import numpy as np
import math
from .hit import Hit

def uniform_charge_cum_current(q, t_start, time_int,kernel):
    tot_c = np.zeros(1600)
    if time_int <= 0:
        return tot_c
    t_start=int(np.floor(t_start))
    time_int=int(np.floor(time_int))
    # Ensure time_int is treated correctly
    dt = time_int
    dq = q / dt
    
    kernel_resp = kernel[kernel != 0]
    kernel_len = len(kernel_resp)
    current = np.zeros(kernel_len+time_int)

    c = kernel_resp * dq

    for i in range(dt):
        current[i:i+kernel_len] += c
    c_cum=np.cumsum(current)
    start=max(0,t_start-kernel_len)
    end=min(len(tot_c),t_start-kernel_len+len(current))
    tot_c[start:end]=c_cum[len(current)-(end-start):]
    if end<len(tot_c): tot_c[end:]=c_cum[-1]
    return tot_c


def modify_signal(signal, window_size=28):
    modified_signal = signal.copy()
    i = 0
    while i < len(signal):
        # Find the first non-zero index
        non_zero_indices = np.where(modified_signal[i:] > 0)[0]
        if len(non_zero_indices) == 0:
            break  # No more non-zero values
        start_index = i + non_zero_indices[0]
        
        # Define the window for averaging
        end_index = min(start_index + window_size, len(signal))
        non_zero_values = modified_signal[start_index:end_index]
        
        # Average all non-zero values within the window
        avg_value = np.mean(non_zero_values[non_zero_values > 0])
        
        # Set the window to the averaged value
        modified_signal[start_index:end_index] = avg_value
        
        # Move to the next non-zero index after the current window
        i = end_index
    
    return modified_signal

def make_dense_WF(arr):
    dense_arr = np.zeros(1600)
    for i in arr:
        dense_arr[int(i.start_time):int(i.end_time)]=i.charge
    return dense_arr
        

def simActivity(chages,kernel_middle,kernel_adj):
    
    return None
    
    
