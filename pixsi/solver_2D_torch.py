import torch
import numpy as np
from .util import uniform_charge_cum_current_part as current_part


# Wrapper that calls current_part and converts its output to a torch tensor on the GPU.
def current_part_torch(s_value, s_dt, kernel, device):
    np_arr = current_part(s_value, s_dt, kernel)
    return torch.tensor(np_arr, device=device, dtype=torch.float32)

def solver_2D_torch(measurements, signals, response, device='cuda'):
    # -------------------------------
    # 1. Prepare Measurement Data on GPU
    # -------------------------------
    # Build measurement mapping: per pixel, store sorted measurement times as GPU tensors.
    # Also build:
    # - samples: mapping (pixel, time) -> measurement value.
    # - meas_key_to_idx: mapping (pixel, time) -> index into valid_measurements.
    measurement_times = {}   # pixel -> list of times (to be converted)
    samples = {}             # (pixel, time) -> measurement value
    meas_key_to_idx = {}     # (pixel, time) -> index into valid_measurements
    valid_measurements = []  # List of valid measurement tuples

    for m in measurements:
        pixel, time, value = m[0], m[1], m[2]
        if value == 5000 or value == 0:
            continue
        key = (pixel, time)
        if pixel not in measurement_times:
            measurement_times[pixel] = []
        measurement_times[pixel].append(time)
        samples[key] = value
        meas_key_to_idx[key] = len(valid_measurements)
        valid_measurements.append(m)
    # Convert each pixel's measurement times to a sorted GPU tensor.
    for pixel in measurement_times:
        sorted_times = sorted(measurement_times[pixel])
        measurement_times[pixel] = torch.tensor(sorted_times, device=device, dtype=torch.float32)

    # -------------------------------
    # 2. Process Signals and Compute Contributions
    # -------------------------------
    # Sort signals by start time (index 3)
    signals_sorted = sorted(signals, key=lambda s: s[3])
    
    # sample_param_map will store contributions: key (pixel, time) -> list of (signal id, contribution)
    sample_param_map = {}
    
    # Determine kernel lengths by converting one kernel via current_part_torch.
    dummy_self = current_part_torch(1.0, 1.0, response[0], device)
    dummy_neighbor = current_part_torch(1.0, 1.0, response[1], device)
    L_self = dummy_self.shape[0]
    L_neighbor = dummy_neighbor.shape[0]
    
    # Dictionaries to hold per-signal state.
    signal_stop_self = {}      # s_id -> stop time for self contributions
    signal_stop_neighbor = {}  # s_id -> dict(pixel -> stop time)
    signal_remaining_self = {}      # s_id -> remaining kernel charge (self)
    signal_remaining_neighbor = {}  # s_id -> dict(pixel -> remaining kernel charge) for neighbors

    # Process contributions for a given pixel and time range (all on GPU)
    def process_contributions(pixel, start_time, end_time, is_self, s_id, conv_self, conv_neighbor):
        if pixel not in measurement_times:
            return
        times = measurement_times[pixel]  # GPU tensor (sorted)
        # Use torch.searchsorted to find the starting index.
        start_time_tensor = torch.tensor(start_time, device=device, dtype=times.dtype)
        start_idx_tensor = torch.searchsorted(times, start_time_tensor)
        start_idx = int(start_idx_tensor.item())
        length = conv_self.shape[0] if is_self else conv_neighbor.shape[0]
        # Get the appropriate remaining_charge and stop_time.
        if is_self:
            remaining_charge = signal_remaining_self[s_id]
            stop_time = signal_stop_self[s_id]
        else:
            remaining_charge = signal_remaining_neighbor[s_id][pixel]
            stop_time = signal_stop_neighbor[s_id][pixel]
        # Loop over measurements from start_idx.
        for i in range(start_idx, times.shape[0]):
            meas_time_tensor = times[i]
            meas_time = meas_time_tensor.item()
            if meas_time > 1600:
                break
            if stop_time is not None and meas_time > stop_time:
                break
            # Compute kernel index by linear scaling.
            ratio = (meas_time_tensor - start_time) / (end_time - start_time) * length
            idx = int(ratio.item())
            idx = min(idx, length - 1)
            Q_contrib = remaining_charge[idx]
            if Q_contrib > 0:
                key = (pixel, meas_time)
                if key not in sample_param_map:
                    sample_param_map[key] = []
                sample_param_map[key].append((s_id, Q_contrib.item()))
                # Only subtract if measurement is valid.
                if samples[key] != 5000 and samples[key] != 0:
                    remaining_charge[idx:] = remaining_charge[idx:] - Q_contrib
                if torch.all(remaining_charge <= 1e-6):
                    if is_self:
                        signal_stop_self[s_id] = meas_time
                    else:
                        signal_stop_neighbor[s_id][pixel] = meas_time
                    break

    # Loop over each signal.
    for s in signals_sorted:
        s_id, s_pixel, s_value, s_t_start, s_dt, flag = s
        # Get convolution kernels by calling current_part_torch (which wraps the original current_part).
        conv_self = current_part_torch(s_value, s_dt, response[0], device)
        conv_neighbor = current_part_torch(s_value, s_dt, response[1], device)
        
        # Define time ranges for self and neighbor contributions.
        start_time_self = max(0, s_t_start - L_self)
        end_time_self = s_t_start + s_dt
        start_time_neighbor = max(0, s_t_start - L_neighbor)
        end_time_neighbor = end_time_self

        # Initialize kernel state.
        signal_stop_self[s_id] = None
        signal_remaining_self[s_id] = conv_self.clone()
        signal_stop_neighbor[s_id] = {}
        signal_remaining_neighbor[s_id] = {}
        neighbors = [s_pixel - 1, s_pixel + 1]
        for neighbor in neighbors:
            signal_stop_neighbor[s_id][neighbor] = None
            signal_remaining_neighbor[s_id][neighbor] = conv_neighbor.clone()
        
        # Process self and neighbor contributions.
        process_contributions(s_pixel, start_time_self, end_time_self, True, s_id, conv_self, conv_neighbor)
        for neighbor in neighbors:
            process_contributions(neighbor, start_time_neighbor, end_time_neighbor, False, s_id, conv_self, conv_neighbor)
    
    # -------------------------------
    # 3. Convert Mapping to a Sparse Tensor on GPU
    # -------------------------------
    N_measurements = len(valid_measurements)
    N_signals = len(signals)
    
    # Count total contributions (only for valid measurement keys).
    total_contribs = 0
    for key, contrib_list in sample_param_map.items():
        if key in meas_key_to_idx:
            total_contribs += len(contrib_list)
    
    # Preallocate GPU tensors for indices and values.
    indices_tensor = torch.empty((2, total_contribs), dtype=torch.long, device=device)
    values_tensor = torch.empty(total_contribs, dtype=torch.float32, device=device)
    
    counter = 0
    for key, contrib_list in sample_param_map.items():
        if key not in meas_key_to_idx:
            continue
        m_idx = meas_key_to_idx[key]
        for (s_id, frac) in contrib_list:
            indices_tensor[0, counter] = m_idx  # measurement index
            indices_tensor[1, counter] = s_id   # signal index
            values_tensor[counter] = frac
            counter += 1
            
    A = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(N_measurements, N_signals))
    
    actual_measurements = torch.tensor([m[2] for m in valid_measurements],
                                         device=device, dtype=torch.float32)
    
    # -------------------------------
    # 4. Optimization with LBFGS on GPU
    # -------------------------------
    initial_guess = [1.0 if s[5] else 0.0 for s in signals]
    params = torch.tensor(initial_guess, device=device, dtype=torch.float32, requires_grad=True)
    
    def objective_function_torch(params):
        predicted = torch.sparse.mm(A, params.unsqueeze(1)).squeeze(1)
        loss = torch.sum((actual_measurements - predicted) ** 2)
        return loss

    optimizer = torch.optim.LBFGS([params],
                                  max_iter=10000,
                                  tolerance_grad=1e-9,
                                  tolerance_change=1e-9,
                                  line_search_fn="strong_wolfe")
    
    def closure():
        optimizer.zero_grad()
        loss = objective_function_torch(params)
        loss.backward()
        return loss

    optimizer.step(closure)
    
    optimized_params = params.detach().cpu().numpy()
    final_signals = [(s[0], s[1], s[2] * q, s[3], s[4]) for s, q in zip(signals, optimized_params)]
    return final_signals, A

# Example usage:
# final_signals, sparse_contrib_matrix = solver_2D_torch(measurements, signals, response)

