import numpy as np
from scipy.interpolate import interp1d


def getKernel_NDLar(path):
    nddata=np.load("response_38_v2b_50ns_ndlar.npy")
    res= []
    
    for i in range(5):
        line=[]
        for j in range(5):
            max_index = np.argmax(nddata[i*10,j*10,:])
            saved = nddata[i*10,j*10,max_index-500:]
            saved = saved[saved!=0] if (i,j)==(0,0) else saved[saved>0]
            line.append(saved)
        res.append(line)
    return res

def getKernel(path,fr_time_tick=0.05,det_time_tick=0.1):
    #currently supports specific format. Needs generalization
    current = np.load(path)['pixel'][191-3]
    if fr_time_tick==det_time_tick:
        current_sampled = current
    else:
        time_05 = np.arange(0, len(current) * fr_time_tick, fr_time_tick)[:-1]
        time_1 = np.arange(0, len(current) * fr_time_tick, det_time_tick)
        interpolator = interp1d(time_05, current, kind='linear')
        current_sampled = interpolator(time_1)#[:1600]
    #modification to fit current simulation constraints
    nonzeroIdxes = np.nonzero(current_sampled)[0]
    res=np.zeros(1600)
    res[:60]=current_sampled[nonzeroIdxes[-1]-60:nonzeroIdxes[-1]]
    return res
    #nonzeroIdxes = np.nonzero(current_sampled)[0]

def getKernel_Ind(path,fr_time_tick=0.05,det_time_tick=0.1):
    #currently supports specific format. Needs generalization
    current = np.load(path)['pixel'][191-9]
    
    if fr_time_tick==det_time_tick:
        current_sampled= current
    else:
        time_05 = np.arange(0, len(current) * fr_time_tick, fr_time_tick)[:-1]
        time_1 = np.arange(0, len(current) * fr_time_tick, det_time_tick)
        interpolator = interp1d(time_05, current, kind='linear')
        current_sampled = interpolator(time_1)#[:1600]
    #modification to fit current simulation constraints
    nonzeroIdxes = np.nonzero(current_sampled)[0]
    res=np.zeros(1600)
    res[:100]=current_sampled[nonzeroIdxes[-1]-100:nonzeroIdxes[-1]]
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
