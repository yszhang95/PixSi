import numpy as np
import math

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

def sim_MIP(t_start,x_start,length,angle):
    t_small = 0.1 # us
    z_small = 0.16 # mm
    dl_small = z_small/math.sin(angle)
    dl_pix = z_small/math.cos(angle)
    dq_small = dl_small*5000 # 5000 e/mm MIP
    t_tmp = t_start
    l_swich =0
    pixel = np.zeros(0,160,1600)
    skipped_pix=int(np.ceil(x_start/4.))
    if skipped_pix>0:
        track=np.array([np.zeros(0,160,1600) for i in range(skipped_pix)])
    else:
        track=[]
    l_tmp = x_start-skipped_pix*4.
    l_tot=0
    while l_tmp<lenght*math.cos(angle):
        if l_tmp>=4.:
            track.append(pixel)
            pixel = np.zeros(0,160,1600)
        pixel[t_tmp]=dq_small
        t_tmp+=1
        l_tmp+=dl_pix
        l_tot+=dl_pix
    return track
        
    
