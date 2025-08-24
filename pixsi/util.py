import numpy as np
import math
from .hit import Hit
from .config import SHORT_HIT , LONG_HIT , INTERMEDIATE_HIT , SHORT_HIT_tick , LONG_HIT_tick , INTERMEDIATE_HIT_tick



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


def extract_TRED_test(file_name):
    key_meas = 'hits_tpc0_batch11'
    key_meas_loc = 'hits_tpc0_batch11_location'
    key_tru = 'effq_tpc0_batch11'
    key_tru_loc = 'effq_tpc0_batch11_location'
    key_wf = 'current_tpc0_batch11'
    key_wf_loc = 'current_tpc0_batch11_location'
    
    #key_meas = 'hits_tpc5_batch0'
    #key_meas_loc = 'hits_tpc5_batch0_location'
    #key_tru = 'effq_tpc5_batch0'
    #key_tru_loc = 'effq_tpc5_batch0_location'
    #key_wf = 'current_tpc5_batch0'
    #key_wf_loc = 'current_tpc5_batch0_location'
    
    with np.load(file_name, allow_pickle=True) as data:
        if key_meas in data:
            tru = data[key_tru].copy()  # .copy() to keep it after closing
            tru_loc = data[key_tru_loc].copy()
            meas = data[key_meas].copy()  # .copy() to keep it after closing
            meas_loc = data[key_meas_loc].copy()
            wf = data[key_wf].copy()
            wf_loc =data[key_wf_loc].copy()
        else:
            raise KeyError(f"Key '{key_meas}' not found in the file.")
    def getHits(arr,arr_loc,state="Hit"):
        pixels = []
        tot=0;
        for n,i in enumerate(arr_loc):
            pixel = (i[0],i[1])
            #print(pixel)
            if state=="Hit":
                if np.isinf(arr[n][3]) or arr[n][3]<0: continue
                pixels.append((pixel,i[2],arr[n][3]))
                tot+=arr[n][3]
            else:
                pixels.append((pixel,i[2],arr[n][0][0]))
        return pixels,tot
    m,_ = getHits(meas,meas_loc)
    print("True Extraction")
    t,_ = getHits(tru,tru_loc)
    w,_ = getHits(wf,wf_loc,"WF")
    return m,t,w


import itertools
# Input formats
# measurements = [((y, z), time, charge), ...]
# signals = [(s_id, (y, z), charge, start_time, end_time, threshold), ...]
# true_charges = [((y, z), time, charge), ...]


import numpy as np

def generate_hits_from_true(true_map, measurement_map, response, interval_short=32,interval_long=56, max_time=12000):
    hits_per_pixel = {}
    len_response = len(response)

    for pixel in true_map:
        effq_list = true_map.get(pixel, [])
        meas_list = measurement_map.get(pixel, [])

        if not effq_list:
            continue  # skip pixels with no true charge

        raw_current = np.zeros(max_time)

        # Step 1: Add scaled response to raw_current
        for t, q in effq_list:
            start = max(0, t-len_response)
            end = t
            r_start = 0+len_response-(end-start)
            raw_current[start:end] += q * response[r_start:]

        # Step 2: Cumulative current
        cumulative = np.cumsum(raw_current)
        #if pixel!=(84, 2): continue
        #import matplotlib.pyplot as plt
        #plt.plot(raw_current)
        #npe=np.array(effq_list)
        #plt.plot(npe[:,0],npe[:,1])
        #plt.plot(effq_list)
        #plt.show()
        hits = []

        if meas_list:
            # Step 3: Measurements present
            meas_times_sorted = sorted(t for t, _ in meas_list)
            last_meas=None
            for t in meas_times_sorted:
                if last_meas is None or t-last_meas>interval_long:
                    interval=interval_short
                    t0 = t
                    t1 = t + interval
                else:
                    interval=interval_long
                    t0 = t-INTERMEDIATE_HIT_tick
                    t1 = t0+interval
                last_meas=t0
                if t1 >= max_time:
                    continue

                q_meas = cumulative[t1]# - cumulative[t0]
                if q_meas <= 0:
                    continue

                hit = {
                    'start_time': t0,
                    'end_time': t1,
                    'charge': q_meas / interval
                }
                hits.append(hit)

                # Subtract charge from future
                cumulative[t1:] -= q_meas

            # Final leftover hit if needed
            last_meas_end = meas_times_sorted[-1] + interval
            remaining_charge = cumulative[-1] - cumulative[last_meas_end] if last_meas_end < max_time else 0
            last_start = min(last_meas_end, max_time - interval)

            if remaining_charge > 0:
                final_hit = {
                    'start_time': last_start,
                    'end_time': last_start + interval,
                    'charge': remaining_charge / interval
                }
                hits.append(final_hit)

        else:
            # Step 4: No measurements — create fallback hit
            total_charge = cumulative[-1]
            if total_charge > 0:
                hit = {
                    'start_time': 0,
                    'end_time': interval_short,
                    'charge': total_charge / interval_short
                }
                hits.append(hit)

        hits_per_pixel[pixel] = hits

    return hits_per_pixel




def create_hits(measurements, signals, true_charges,tpc_id,event_id,response,time_tick=0.05):
    short_hit=SHORT_HIT_tick #int(SHORT_HIT/time_tick)
    long_hit =LONG_HIT_tick # int(LONG_HIT/time_tick)
    intermediate_hit = INTERMEDIATE_HIT_tick #int(INTERMEDIATE_HIT/time_tick)
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
        true_by_pixel[pixel].append((time, charge))
        seen += charge
    
    
    true_hit_perpix = generate_hits_from_true(true_by_pixel, meas_by_pixel, response, interval_short=short_hit,interval_long=long_hit, max_time=12000)
    #for i in true_by_pixel:
    #    tot_charge = sum([c[2] for c in true_by_pixel[i]])
    #    true_hit_perpix.append(Hit(0, 0, 0, i, 'true fake', tot_charge, 0, 1))
    cnt_neg=1
    for pixel,hits in true_hit_perpix.items():
        for h in hits:
            recorded+=h['charge']*(h['end_time']-h['start_time'])
            if (pixel,h['start_time']) in meas_index :
                id_true=meas_index[(pixel,h['start_time'])]
            else:
                id_true=-1*cnt_neg
                cnt_neg+=1
            true_hits.append(Hit(id_true,tpc_id,event_id,pixel,'true',h['charge'],h['start_time'],h['end_time']))
    
    eff_hits = []
    for k,v in true_by_pixel.items():
        for h in v:
            eff_hits.append(Hit(0,0,0,k,'true',h[1],h[0],h[0]))
    
    # For each pixel, compute overlaps of true hits with measurement intervals
    #for pixel, intervals in meas_intervals.items():
    #    for hid, start, end in intervals:
    #        total_overlap_charge = 0.0
    #        for true_start, true_end, charge_per_tick in true_by_pixel.get(pixel, []):
    #            overlap_start = max(start, true_start)
    #            overlap_end = min(end, true_end)
    #            if overlap_start < overlap_end:
    #                overlap_len = overlap_end - overlap_start
    #                total_overlap_charge += charge_per_tick * overlap_len
    #        interval_len = end - start
    #        norm_charge = total_overlap_charge / interval_len if interval_len > 0 else 0.0
            #true_hits.append(Hit(hid, tpc_id, event_id, pixel, 'true', norm_charge, start, end))
            #recorded += total_overlap_charge
            
    print("True Charge from TRED: ",seen)
    print("True Charge recorded in hits: ",recorded)
    return meas_hits , signal_hits , true_hits , true_hit_perpix , eff_hits


