import numpy as np
from .kernels import *
import math
from .util import extract_TRED_test as readTredInput
from tqdm import tqdm

def burstMode(file_name):
    m,t,w = readTredInput(file_name)
    bm=[]
    dt_sample = 7
    dt = 16

    for pixel in tqdm(w):
        w_int = np.cumsum(pixel[2])

        # Find first index where cumulative sum exceeds 5000
        idx = np.searchsorted(w_int, 5)
        # Make sure we have enough values for the range we're going to sample
        if idx + dt * dt_sample >= len(w_int):
            continue  # skip if not enough data
        # Vectorized append (could use preallocation if needed)
        bm.extend(
            (pixel[0], idx + dt * s, w_int[idx + dt * s])
            for s in range(dt_sample)
        )
    
    print(t[:3])
    return bm , t , w

def current(q, z):
    c = kernel() * q
    shift = int((z - 10) / 1.6 / 0.1)
    c2 = np.zeros(len(c))
    c2[shift:] = c[:len(c) - shift]
    return c2

def qum_current(q, z):
    q_sum = 0
    c = current(q, z)
    res = []
    for i in c:
        q_sum += i
        res.append(q_sum)
    return res

def compute_current(signal, kernel, time_steps):
    kernel = kernel[kernel != 0]
    kernel_len = len(kernel)
    current = np.zeros(len(time_steps))
    for t, charge in enumerate(signal):
        if charge > 0:
            current[max(0,t - kernel_len + 1) : t + 1] += charge * kernel[len(kernel)-(t+1-max(0,t - kernel_len + 1)):]
    return current
    
def trigger_legacy(c,trsh=200):
         # with 90 large tail
    read=int(1.5/0.1)+1
    dead=1
    hold=int(1.1/0.1)+dead
    meas=[]
    qsum=0
    busy=False
    dead=False
    n=0
    actualq=[]
    while n<len(c):
        if not dead:
            qsum+=c[n]
        else:
            dead=False
            busy=True
        if qsum>=trsh and not busy:
            time=n
            reading=0
            while reading<read:
                n+=1
                reading+=1
                if n<len(c): qsum+=c[n]
                actualq.append(qsum)
            meas.append((time,qsum))
            qsum=0
            dead=True
        else:
            actualq.append(qsum)
            n+=1
        if busy:
            hold-=1
        if hold<=0:
            busy=False
            hold=int(1.1/0.1)+dead
    return meas,actualq
    
def trigger(c,trsh=200):
         # with 90 large tail
    read=int(1.5/0.1)+1
    dead=1
    hold=int(1.1/0.1)+dead
    meas=[]
    qsum=0
    busy=False
    dead=False
    n=0
    actualq=[]
    while n<len(c):
        if not dead:
            qsum+=c[n]
        else:
            dead=False
            busy=True
        if qsum>=trsh and not busy:
            time=n
            reading=0
            while reading<read:
                n+=1
                reading+=1
                if n<len(c): qsum+=c[n]
                actualq.append(qsum)
            meas.append((time,qsum))
            qsum=0
            dead=True
        else:
            actualq.append(qsum)
            n+=1
        if busy:
            hold-=1
            if qsum >= trsh and hold > 0:  # Check if threshold is crossed within hold period
                reading = 0
                while reading < (read + hold):  # Extend accumulation to 27 ticks
                    n += 1
                    reading += 1
                    if n < len(c):qsum += c[n]
                    actualq.append(qsum)
                meas.append((n-16, qsum))  # Store measurement at 11 ticks after dead time
                qsum = 0
                dead = True
                busy = True
                hold=int(1.1/0.1)+dead
            elif hold<=0:
                busy=False
                hold=int(1.1/0.1)+dead
    return meas,actualq


def sim_MIP(t_start,x_start,length,angle):
    t_small = 0.1 # us
    z_small = 0.16 # mm
    angle_radians = math.radians(angle)
    dl_small = z_small/math.sin(angle_radians)
    dl_pix = dl_small*math.cos(angle_radians)
    dq_small = dl_small*5000 # 5000 e/mm MIP
    t_tmp = t_start
    pixel = np.zeros(1600)
    skipped_pix=int(np.ceil(x_start/4.))
    if skipped_pix>0:
        track=[np.zeros(1600) for i in range(skipped_pix)]
    else:
        track=[]
    l_tmp = max(0,x_start-skipped_pix*4.)
    l_tot=0
    while l_tot<length*math.cos(angle_radians) and t_tmp<1600:
        if l_tmp>=4.:
            track.append(pixel)
            pixel = np.zeros(1600)
            l_tmp=0
        pixel[t_tmp]=dq_small
        #print(l_tmp)
        t_tmp+=1
        l_tmp+=dl_pix
        l_tot+=dl_pix
    if len(np.nonzero(pixel)[0])>0:track.append(pixel)
    return track

def simActivity_toy(pixels,kernel_middle,kernel_adj):
    time_steps=np.linspace(0, 1600, 1600)
    currents = np.array([np.zeros(1600) for _ in range(len(pixels)+2)])
    for p in range(1,len(pixels)+1):
        current_response_middle = compute_current(pixels[p-1], kernel_middle, time_steps)
        current_response_side = compute_current(pixels[p-1], kernel_adj, time_steps)
        currents[p-1]+=current_response_side
        currents[p]+=current_response_middle
        currents[p+1]+=current_response_side
    
    import matplotlib.pyplot as plt
    #for n,i in enumerate(pixels):
    #    plt.plot(i,label="charge %.2f"%n)
    #for n,i in enumerate(currents):
    #    plt.plot(np.cumsum(i),label="current %.2f"%n)
    
    trsh=5000
    from .preproc import process_measurements as PreProc
    from .hit import Hit
    measurements = []
    blocks = []
    hits=[]
    for nc,c in enumerate(currents):
        if np.sum(c)<trsh:
            #measurements.append([])
            continue
        M,q=trigger(c,trsh)
        #plt.plot(q,label="actual current %.2f"%nc)
        M_blocks=PreProc(M,trsh)
        for i in M:
            measurements.append((nc,i[0],i[1]))
        blocks.append(M_blocks)
        trueHits=[]
        rawHits=[]
        #Create True Hits
        for block in M_blocks:
            true_times=np.nonzero(pixels[nc-1])[0]
            if len(true_times)==0 : continue
            chg=true_times[0]
            tr=block[0][0]
            kl=len(kernel_middle)
            n = len(block)-1
            #slices = [(chg,tr+kl)]+[(tr+kl,tr+kl+16)]+[(tr+kl+16+28*i,tr+kl+16+28*(i+1)) for i in range(n-2)]
            slices = [(chg,tr+16)]+[(tr+16+28*i,tr+16+28*(i+1)) for i in range(n-1)] #for simple short hit defenition
            #print(slices)
            for s in slices:
                dt_true = s[1]-s[0]
                if dt_true==0: continue
                avg=np.sum(pixels[nc-1][slice(s[0],s[1])])/dt_true
                if avg==0: continue
                h = Hit(avg,s[0],s[1])
                trueHits.append(h)
            pixels[nc-1][chg:slices[-1][1]]=0
        #Create Raw Hits
        for block in M_blocks:
            #print("debug: ",block)
            for n,m in enumerate(block):
                if n==0 or n==len(block)-1:
                    continue
                if n==1:
                    h=Hit(m[1]/16,m[0]-16,m[0])
                else:
                    h=Hit(m[1]/28,m[0]-28,m[0])
                rawHits.append(h)
        hits.append([nc,[trueHits,rawHits]])
    #plt.legend()
    #plt.show()
    
    return measurements,blocks,hits
    


