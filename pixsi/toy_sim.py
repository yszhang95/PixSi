import numpy as np
from .kernels import *
import math

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
        if hold<=0:
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
