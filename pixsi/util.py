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
    
    
def uniform_charge_cum_current_part(q,time_int,kernel):
    kernel_resp = kernel[kernel != 0]
    kernel_len = len(kernel_resp)
    if time_int <= 0:
        return np.zeros(kernel_len)
    time_int=int(np.floor(time_int))
    # Ensure time_int is treated correctly
    dt = time_int
    dq = q / dt
    
    current = np.zeros(kernel_len+time_int)

    c = kernel_resp * dq
    
    for i in range(dt):
        current[i:i+kernel_len] += c
        
    return np.cumsum(current)


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
        

import numpy as np
from sortedcontainers import SortedList
from collections import defaultdict

def build_signal_measurement_map(measurements, signals, kernel_length_mid, kernel_length_ind):
    """
    Build a mapping from measurement indices to indices of signals contributing to it.
    
    Parameters:
    - measurements: List of tuples (pixelID, measurement_value, time)
    - signals: List of tuples (spID, pixelID, signal, t_start, delta_t)
    - kernel_length_mid: Influence duration for the same pixel
    - kernel_length_ind: Influence duration for neighboring pixels
    
    Returns:
    - measurement_signal_map: Dictionary {measurement_index: [signal_indices]}
    """
    
    # Sort signals by start time to allow efficient searching
    signals = sorted(signals, key=lambda s: s[3])  # Sort by t_start
    signal_start_times = np.array([s[3] for s in signals])  # t_start values for fast lookup
    signal_map = defaultdict(list)  # Measurement index → List of signal indices

    # Use SortedList for fast insertion and range search
    active_signals = SortedList()
    
    # Process each measurement and find contributing signals
    for meas_idx, (m_pixel, _, m_time) in enumerate(measurements):
        contributing_signals = []

        # Find signals that could contribute (Binary search for efficiency)
        t_min = m_time - max(kernel_length_mid, kernel_length_ind)  # Earliest time of influence
        t_max = m_time  # Only consider signals that could influence up to measurement time

        start_idx = np.searchsorted(signal_start_times, t_min, side='left')  # Fast lookup
        end_idx = np.searchsorted(signal_start_times, t_max, side='right')

        # Check only relevant signals in range
        for sig_idx in range(start_idx, end_idx):
            spID, s_pixel, signal, t_start, delta_t = signals[sig_idx]

            # Check if signal affects this measurement
            if (s_pixel == m_pixel and t_start - kernel_length_mid <= m_time <= t_start + delta_t) or \
               (abs(s_pixel - m_pixel) == 1 and t_start - kernel_length_ind <= m_time <= t_start + delta_t):
                contributing_signals.append(sig_idx)

        # Store results in mapping
        if contributing_signals:
            signal_map[meas_idx] = contributing_signals

    return signal_map

    
import h5py
from collections import defaultdict

def extract_TRED_by_tpc(file_name):
    def extract_group_data(group):
        # Load datasets
        grid = group['grid_index'][:]
        pos = group['position'][:]
        charge = group['charge'][:]
        tpc_id = group['tpc_id'][:]

        # Prepare: tpc_id -> tuple of arrays
        data_by_tpc = defaultdict(list)
        for tid in np.unique(tpc_id):
            mask = (tpc_id == tid)
            grid_x = grid[mask][:, 0]
            grid_y = grid[mask][:, 1]
            grid_t = grid[mask][:, 2]
            chg = charge[mask]
            res=[]
            for i in range(len(chg)):
                res.append(((grid_x[i],grid_y[i]),grid_t[i],chg[i]))
            data_by_tpc[tid] = res
        
        return data_by_tpc

    with h5py.File(file_name, 'r') as f:
        hits_data = extract_group_data(f['hits'])
        effq_data = extract_group_data(f['effq'])

    return hits_data, effq_data

