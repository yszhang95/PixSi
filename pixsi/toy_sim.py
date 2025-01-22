import numpy as np
from .kernels import kernel

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
            current[t - kernel_len + 1 : t + 1] += charge * kernel
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
