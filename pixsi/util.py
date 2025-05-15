import numpy as np
import math
from .hit import Hit
from .config import SHORT_HIT , LONG_HIT , INTERMEDIATE_HIT



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
        
    return np.array(current)


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
        event_id = group['event_id'][:]

        # Prepare: tpc_id -> tuple of arrays
        data_by_tpc = defaultdict(list)
        for tid in np.unique(tpc_id):
            mask = (tpc_id == tid)
            if tid!=1: continue
            grid_x = grid[mask][:, 0]
            grid_y = grid[mask][:, 1]
            grid_t = grid[mask][:, 2]
            chg = charge[mask]
            event_id=event_id[mask]
            res=[]
            for i in range(len(chg)):
                if event_id[i]!=815: continue
                res.append(((grid_x[i],grid_y[i]),grid_t[i],chg[i]))
            data_by_tpc[tid] = res
        
        return data_by_tpc

    with h5py.File(file_name, 'r') as f:
        hits_data = extract_group_data(f['hits'])
        effq_data = extract_group_data(f['effq'])

    return hits_data, effq_data


import itertools
# Input formats
# measurements = [((y, z), time, charge), ...]
# signals = [(s_id, (y, z), charge, start_time, end_time, threshold), ...]
# true_charges = [((y, z), time, charge), ...]

def create_hits(measurements, signals, true_charges,tpc_id,event_id,time_tick=0.05):
    short_hit=int(SHORT_HIT/time_tick)
    long_hit = int(LONG_HIT/time_tick)
    intermediate_hit = int(INTERMEDIATE_HIT/time_tick)
    # Step 1: Group and process measurements
    meas_by_pixel = defaultdict(list)
    for pixel, time, charge in sorted(measurements, key=lambda x: x[1]):
        meas_by_pixel[pixel].append((time, charge))

    meas_hits = []
    meas_index = {}  # Key: (pixel, time): hit_id
    hit_id_counter = itertools.count()
    meas_intervals = defaultdict(list)

    for pixel, meas_list in meas_by_pixel.items():
        last_time = None
        for time, charge in sorted(meas_list, key=lambda x: x[0]): #meas_list:
            if last_time is None or time - last_time > long_hit:
                interval_len = short_hit
                start_time = time
            else:
                start_time = time - intermediate_hit
                interval_len = long_hit
            end_time = start_time + interval_len
            norm_charge = charge / interval_len
            hid = next(hit_id_counter)

            #pixsi.hit.Hit(hit_ID , tpc_ID , event_ID,pixel_ID, type, charge, start_time, end_time)
            meas_hits.append(Hit(hid,tpc_id,event_id,pixel,'raw',norm_charge,start_time,end_time))
            meas_index[(pixel, start_time)] = hid  # Lookup for signals
            meas_intervals[pixel].append((hid, start_time, end_time)) # Lookup for truth
            last_time = time

    # Step 2: Process signal hits with aligned measurement ID
    signal_hits = []
    #print("measurements = ",measurements)
    #print("meas intervals = ",meas_index)
    #print("signals = ",signals)
    for _, pixel, charge, start_time, dt in signals:
        matching_meas_id = meas_index.get((pixel, start_time))
        assert matching_meas_id is not None, f"No matching measurement for signal at pixel {pixel} and start_time {start_time}"
        signal_hits.append(Hit(matching_meas_id,tpc_id,event_id,pixel,'signal',charge/dt,start_time,start_time+dt))

    # Step 3: Process true hits with uniform 300-tick window logic
    true_hits = []
    true_by_pixel = defaultdict(list)
    seen = 0
    recorded = 0

    # Preprocess true hits per pixel, discard small charges
    for pixel, time, charge in true_charges:
        if abs(charge) < 1e-6:
            continue
        window_start = time -5
        window_end = time + 5  # 300 ticks duration
        charge_per_tick = charge / 10.0
        true_by_pixel[pixel].append((window_start, window_end, charge_per_tick))
        seen += charge
    
    
    true_hit_perpix = []
    for i in true_by_pixel:
        tot_charge = sum([c[2] for c in true_by_pixel[i]])
        true_hit_perpix.append(Hit(0, 0, 0, i, 'true fake', tot_charge*10.0, 0, 1))
        
    
    # For each pixel, compute overlaps of true hits with measurement intervals
    for pixel, intervals in meas_intervals.items():
        for hid, start, end in intervals:
            total_overlap_charge = 0.0
            for true_start, true_end, charge_per_tick in true_by_pixel.get(pixel, []):
                overlap_start = max(start, true_start)
                overlap_end = min(end, true_end)
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    total_overlap_charge += charge_per_tick * overlap_len
            interval_len = end - start
            norm_charge = total_overlap_charge / interval_len if interval_len > 0 else 0.0
            true_hits.append(Hit(hid, tpc_id, event_id, pixel, 'true', norm_charge, start, end))
            recorded += total_overlap_charge
            
    print("True Charge from TRED: ",seen)
    print("True Charge recorded in hits: ",recorded)
    return meas_hits , signal_hits , true_hits , true_hit_perpix


