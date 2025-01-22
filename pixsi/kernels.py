import numpy as np

def kernel():
    drift_speed = 1.6
    sigma = 1.5
    peak_value = 1.0
    drift_time = 5.0 / drift_speed
    time_steps = np.linspace(0, 160, 1600)

    def half_gaussian(t, sigma, peak_value, drift_time):
        return peak_value * np.exp(t**2 / sigma**2) if t <= drift_time else 0

    return np.array([half_gaussian(t, sigma, peak_value, drift_time) for t in time_steps])
