import numpy as np
from scipy.interpolate import interp1d


def getKernel_NDLar(path,kind = "regular"):
    if "shield" in path:
        nddata=np.load(path)['response']
        bin_tick=0.1
    else:
        nddata=np.load(path)
        bin_tick=0.05
    res= []
    import matplotlib.pyplot as plt
    # this segment is for debugging : to apply specific FR instead of the averaged
    if False:
        for i in range(5):
            line=[]
            for j in range(5):
                saved = np.zeros(len(nddata[i*10,j*10,:]))
                saved+=nddata[i*10,j*10,:]*bin_tick
                saved=saved[saved!=0]
                line.append(saved)
            res.append(line)

        return res
    #calculate average FR
    for i in range(5):
        line=[]
        for j in range(5):
            #max_index = np.argmax(nddata[i*10,j*10,:])
            saved = np.zeros(len(nddata[i*10,j*10,:]))
            if i==j==0:
                for k in range(0,5):
                    for l in range(0,5):
                        saved += nddata[i*10+k,j*10+l,:]*bin_tick#max_index-1500:]
                saved/=25.0
            elif i==0 and j!=0:
                for k in range(-5,5):
                    for l in range(0,5):
                        saved += nddata[i*10+k,j*10+l,:]*bin_tick#max_index-1500:]
                saved/=50.0
            elif i!=0 and j==0:
                for k in range(0,5):
                    for l in range(-5,5):
                        saved += nddata[i*10+k,j*10+l,:]*bin_tick#max_index-1500:]
                saved/=50.0
            elif i!=0 and j!=0:
                for k in range(-5,5):
                    for l in range(-5,5):
                        saved += nddata[i*10+k,j*10+l,:]*bin_tick#max_index-1500:]
                saved/=100.0
            saved = saved[saved!=0] #if (i,j)==(0,0) else saved#[saved>0]
            line.append(saved)
        res.append(line)
    print("Res Kernel Shape : ",len(res),len(res[0]))
    #plt.plot(nddata[0,0,:],label=f"path 3,4")
    #plt.plot(nddata[10,10,:],label=f"path 4,3")
    #plt.legend(loc='upper left')
    #plt.plot(res[0][0])
    #plt.show()
    if kind != "regular":
        res_cum = [[None]*5 for _ in range(5)]
        for du in range(5):
            for dv in range(5):
                arr = np.asarray(res[du][dv], dtype=np.float32)
                res_cum[du][dv] = np.cumsum(arr, dtype=np.float64).astype(np.float32)
        return res_cum
    else:
        return res
    
def getKernel_NDLar_withgrid(path):
    nddata=np.load(path)['response']
    res= []
    import matplotlib.pyplot as plt
    for i in range(5):
        line=[]
        for j in range(5):
            saved = np.zeros(len(nddata[i*10,j*10,:]))
            if i==j==0:
                for k in range(0,5):
                    for l in range(0,5):
                        saved += nddata[i*10+k,j*10+l,:]*0.05#max_index-1500:]
                saved/=25.0
            elif i==0 and j!=0:
                for k in range(-5,5):
                    for l in range(0,5):
                        saved += nddata[i*10+k,j*10+l,:]*0.05#max_index-1500:]
                saved/=50.0
            elif i!=0 and j==0:
                for k in range(0,5):
                    for l in range(-5,5):
                        saved += nddata[i*10+k,j*10+l,:]*0.05#max_index-1500:]
                saved/=50.0
            elif i!=0 and j!=0:
                for k in range(-5,5):
                    for l in range(-5,5):
                        saved += nddata[i*10+k,j*10+l,:]*0.05#max_index-1500:]
                saved/=100.0
            saved = saved[saved!=0]
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
