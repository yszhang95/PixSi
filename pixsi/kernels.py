import numpy as np
from scipy.interpolate import interp1d

def getKernel(path):
    #currently supports specific format. Needs generalization
    current = np.load(path)['pixel'][191-3]
    
    time_05 = np.arange(0, len(current) * 0.05, 0.05)[:-1]
    time_1 = np.arange(0, len(current) * 0.05, 0.1)
    interpolator = interp1d(time_05, current, kind='linear')
    current_sampled = interpolator(time_1)#[:1600]
    #modification to fit current simulation constraints
    nonzeroIdxes = np.nonzero(current_sampled)[0]
    res=np.zeros(1600)
    res[:60]=current_sampled[nonzeroIdxes[-1]-60:nonzeroIdxes[-1]]
    return res
    #nonzeroIdxes = np.nonzero(current_sampled)[0]


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

def kernel_1us():
    #1us kernel
    drift_speed = 1.6
    sigma = 0.5
    peak_value = 1.0
    drift_time = 2.0 / drift_speed
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
