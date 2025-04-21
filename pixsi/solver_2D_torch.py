import torch
import numpy as np
from .util import uniform_charge_cum_current_part as current_part


def solver_2D_torch_general(measurements, signals, response, device='cuda'):
    

    def current_part_torch(s_value, s_dt, kernel, device):
        np_arr = current_part(s_value, s_dt, kernel)
        return torch.tensor(np_arr, device=device, dtype=torch.float32)

    measurement_times = {}
    samples = {}
    meas_key_to_idx = {}
    valid_measurements = []

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

    for pixel in measurement_times:
        sorted_times = sorted(measurement_times[pixel])
        measurement_times[pixel] = torch.tensor(sorted_times, device=device, dtype=torch.float32)

    signals_sorted = sorted(signals, key=lambda s: s[3])
    sample_param_map = {}

    R = len(response) - 1
    signal_stop = {}
    signal_remaining = {}

    def process_contributions(s_id, dy, dz, s_pixel, start_time, end_time, conv_q):
        
        neighbor = s_pixel + dy + 10 * dz

        if neighbor not in measurement_times:
            return
            
        times = measurement_times[neighbor]
        start_idx = int(torch.searchsorted(times, torch.tensor(start_time, device=device)).item())
        length = conv_q.shape[0]

        key = (s_id, dy, dz)
        signal_remaining[key] = conv_q.clone()
        signal_stop[key] = None
        remaining_charge = signal_remaining[key]

        for i in range(start_idx, times.shape[0]):
            meas_time = times[i].item()
            if meas_time > 1600:
                break
            if signal_stop[key] is not None and meas_time > signal_stop[key]:
                break

            idx = int(((times[i] - start_time) / (end_time - start_time) * length).item())
            idx = min(idx, length - 1)
            Q_contrib = remaining_charge[idx]

            if Q_contrib > 0:
                meas_key = (neighbor, meas_time)
                if meas_key not in sample_param_map:
                    sample_param_map[meas_key] = []
                sample_param_map[meas_key].append((s_id, Q_contrib.item()))
                if samples[meas_key] not in [0, 5000]:
                    remaining_charge[idx:] -= Q_contrib
                if torch.all(remaining_charge <= 1e-6):
                    signal_stop[key] = meas_time
                    break

    for s in signals_sorted:
        s_id, s_pixel, s_value, s_t_start, s_dt, _ = s
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                kernel = response[abs(dy)][abs(dz)]
                conv_q = current_part_torch(s_value, s_dt, kernel, device)
                start_time = max(0, s_t_start - len(kernel))
                end_time = s_t_start + s_dt
                process_contributions(s_id, dy, dz, s_pixel, start_time, end_time, conv_q)

    # --- Sparse matrix construction and optimization (same as original) ---
    N_measurements = len(valid_measurements)
    N_signals = len(signals)
    total_contribs = sum(len(v) for k, v in sample_param_map.items() if k in meas_key_to_idx)

    indices_tensor = torch.empty((2, total_contribs), dtype=torch.long, device=device)
    values_tensor = torch.empty(total_contribs, dtype=torch.float32, device=device)
    
    counter = 0
    for key, contrib_list in sample_param_map.items():
        if key not in meas_key_to_idx:
            continue
        m_idx = meas_key_to_idx[key]
        for (s_id, frac) in contrib_list:
            indices_tensor[0, counter] = m_idx
            indices_tensor[1, counter] = s_id
            values_tensor[counter] = frac
            counter += 1
            
    A = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(N_measurements, N_signals))
    actual_measurements = torch.tensor([m[2] for m in valid_measurements], device=device, dtype=torch.float32)

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

