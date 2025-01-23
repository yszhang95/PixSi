import numpy as np

def kernel_3us():
    # For 3 us kernel sigma is 1.5 and drft is 5/speed
    drift_speed = 1.6
    sigma = 1.5
    peak_value = 1.0
    drift_time = 5.0 / drift_speed
    time_steps = np.linspace(0, 160, 1600)

    def half_gaussian(t, sigma, peak_value, drift_time):
        return peak_value * np.exp(t**2 / sigma**2) if t <= drift_time else 0

    return np.array([half_gaussian(t, sigma, peak_value, drift_time) for t in time_steps])

def kernel():
    #Default 2us kernel
    drift_speed = 1.6
    sigma = 1.
    peak_value = 1.0
    drift_time = 3.0 / drift_speed
    time_steps = np.linspace(0, 160, 1600)

    def half_gaussian(t, sigma, peak_value, drift_time):
        return peak_value * np.exp(t**2 / sigma**2) if t <= drift_time else 0

    return np.array([half_gaussian(t, sigma, peak_value, drift_time) for t in time_steps])
