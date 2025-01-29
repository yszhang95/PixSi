import sys
import json
import click
import pixsi
from . import units


@click.group()
@click.option("-s","--store",type=click.Path(),
              envvar="PIXSI_STORE",
              help="File for primary data storage (input/output)")
@click.option("-o","--outstore",type=click.Path(),
              help="File for output (primary only input)")
@click.pass_context
def cli(ctx, store, outstore):
    '''
    PixSi command line interface
    '''
    if not store:
        store = "."
    ctx.obj = pixsi.main.Main(store, outstore)


@cli.command()
@click.option("-i","--input", type=str, required=False,
              help="Placeholder")
@click.option("-k","--kernelresp",type=click.Path(),required=False,
              help="Path to Field Response")
@click.pass_context
def run_1D(ctx,input,kernelresp):
    '''
    Runs Toy Simulation + Reconstruction for a single pixel
    '''
    
    if not kernelresp:
        print("Path to FR was not provided. Using Toy Kernel")
        kernel=pixsi.toy_sim.kernel()
    else:
        kernel=pixsi.toy_sim.getKernel(kernelresp)
    import matplotlib.pyplot as plt
    import numpy as np
    #define example true signal
    time_steps = np.linspace(0, 160, 1600)
    true_signal = np.zeros(len(time_steps))
    true_signal[63:62+5*9+1:5] = 1.0  # Example true signal
    true_signal[90:93] = 3.0
    true_signal[93:94] = 7.0
    true_signal[94:95] = 2.0# Example true signal
    #true_signal[187:187+4+1] = 1.0
    plt.plot(time_steps,kernel*100)
    plt.plot(time_steps,np.cumsum(kernel*100))
    plt.title("Current Kernel")
    plt.xlabel("Time (us)")
    #plt.xlim(0,5)
    plt.show()
    #Get example induced Current
    current_response = pixsi.toy_sim.compute_current(true_signal, kernel, time_steps)
    
    
    trsh=0.001
    #Get example sparce triggered mesurements (time,value)
    M,q=pixsi.toy_sim.trigger(current_response,trsh)
    b=np.array(M)
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, true_signal, label="True Signal (Charge)", linestyle="--")
    plt.plot(time_steps, current_response, label="True Current (Current)")
    plt.scatter(b[:,0]*0.1,b[:,1], label="True Measurement")
    plt.xlim(0,13)
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Signal Reconstruction from Current Measurements")
    plt.show()
    plt.show()
    #Create dense measurement across full time interval (Not used)
    m_dense=np.zeros(len(time_steps))
    for i in M:
        m_dense[i[0]]=i[1]
    
    #Devide Measurements into blocks
    M_blocks=pixsi.preproc.process_measurements(M,trsh)
    
    #Fit each block
    RecoveredSignals=[]
    for n,block in enumerate(M_blocks):
        result = pixsi.solver.solver_scipy(block,kernel) #'L-BFGS-B' , SLSQP
        RecoveredSignals.append(result)
    
    
    #Plot Results
    lk=len(kernel[kernel!=0])
    print(RecoveredSignals,lk)
    Recovedred_dense=np.zeros(1600)
    t_st=int(RecoveredSignals[0][0])
    Q=RecoveredSignals[0][1:]
    dt1=M_blocks[0][0][0]+lk-t_st
    dt2=M_blocks[0][1][0]-M_blocks[0][0][0]
    dt3=M_blocks[0][2][0]-M_blocks[0][1][0]
    Recovedred_dense[t_st:M_blocks[0][0][0]+lk]=Q[0]/dt1
    Recovedred_dense[M_blocks[0][0][0]+lk:M_blocks[0][1][0]+lk]=Q[1]/dt2
    Recovedred_dense[M_blocks[0][1][0]+lk:M_blocks[0][2][0]+lk]=Q[2]/dt3
    print("Time Intervals: ",dt1,dt2,dt3)
    print("True signal=",np.sum(true_signal[:150]))
    print("Recovered signal=",np.sum(Recovedred_dense[:150]))
    t1=np.linspace(0,160,1600)
    plt.plot(t1,Recovedred_dense,label="Recovered Signal ")
    plt.plot(time_steps, true_signal*1, label="True Signal (Charge)", linestyle="--")
    
    modified_signal = np.zeros(len(true_signal))
    dt1_true=M_blocks[0][0][0]+lk-63
    dt2_true=16
    dt3_true=28
    avg1=np.sum(true_signal[63:M_blocks[0][0][0]+lk])/dt1_true
    avg2=np.sum(true_signal[M_blocks[0][0][0]+lk:M_blocks[0][1][0]+lk])/dt2_true
    avg3=np.sum(true_signal[M_blocks[0][1][0]+lk:M_blocks[0][2][0]+lk])/dt3_true if dt3_true>0 else 0
    modified_signal[63:63+dt1_true]=avg1
    modified_signal[63+dt1_true:63+dt1_true+dt2_true]=avg2
    modified_signal[63+dt1_true+dt2_true:63+dt1_true+dt2_true+dt3_true]=avg3
    print("Expectation: ",np.array([63,avg1*dt1_true,avg2*dt2_true,avg3*dt3_true]))
    plt.plot(time_steps, modified_signal, label="True Signal (Modified)", linestyle="--")
    plt.legend()
    plt.plot()
    plt.xlim(4,13)
    plt.ylim(0,1.5)
    plt.show()

@cli.command()
@click.option("-i","--input", type=str, required=False,
              help="Placeholder")
@click.option("-k","--kernelresp",type=click.Path(),required=False,
              help="Path to Field Response")
@click.pass_context
def run_2D(ctx,input,kernelresp):
    '''
    Runs Toy Simulation + Reconstruction for two MIP tracks in pseudo 2D
    '''
    if not kernelresp:
        print("Path to FR was not provided. Using Toy Kernel")
        kernel=pixsi.toy_sim.kernel()
    else:
        kernel=pixsi.toy_sim.getKernel(kernelresp)*100
    kernel=kernel[kernel!=0]
    import matplotlib.pyplot as plt
    import numpy as np
    #define example true signal
    track1 = pixsi.toy_sim.sim_MIP(150,0,80,45)
    track2 = pixsi.toy_sim.sim_MIP(190,0,80,10)
    print("Total Track1 Charge: ",np.sum(np.array([np.sum(i) for i in track1])))
    if len(track2)>0: print("Total Track2 Charge: ",np.sum(np.array([np.sum(i) for i in track2])))
    if len(track2)>0 and len(track1)<len(track2):
        arr = [np.zeros(1600) for _ in range(len(track2)-len(track1))]
        track1=track1+arr
    pixels = np.array([a + b for a, b in zip(track1, track2)]) if len(track2)>0 else np.array(track1)
    print("Used Pixels: ",len(pixels))
    FinalHits = []
    for pn,p in enumerate(pixels):
        #if pn>0:
        #    continue
        trueHits=[]
        rawHits=[]
        spHits=[]
        #Simulate raw measurements
        trsh=5000
        if sum(p)==0:
            continue
        time_steps=np.linspace(0, 1600, 1600)
        current_response = pixsi.toy_sim.compute_current(p, kernel, time_steps)
        M,q=pixsi.toy_sim.trigger(current_response,trsh)
        #plt.plot(time_steps,p,label="Charge")
        bb=np.array(M)
        #plt.plot(time_steps,current_response,label="Current")
        #plt.scatter(bb[:,0],bb[:,1],label="Measurement")
        #plt.legend()
        #plt.show()
        #Devide Measurements into blocks
        M_blocks=pixsi.preproc.process_measurements(M,trsh)
        #Create True Hits
        for block in M_blocks:
            true_times=np.nonzero(p)[0]
            if len(true_times)==0 : continue
            chg=true_times[0]
            tr=block[0][0]
            kl=len(kernel)
            n = len(block)-1
            slices = [(chg,tr+kl)]+[(tr+kl,tr+kl+16)]+[(tr+kl+16+28*i,tr+kl+16+28*(i+1)) for i in range(n-2)]
            for s in slices:
                dt_true = s[1]-s[0]
                if dt_true==0: continue
                avg=np.sum(p[slice(s[0],s[1])])/dt_true
                h = pixsi.hit.Hit(avg,s[0],s[1])
                trueHits.append(h)
            p[chg:slices[-1][1]]=0
        #Create Raw Hits
        for block in M_blocks:
            for n,m in enumerate(block):
                if n==0 or n==len(block)-1:
                    continue
                if n==1:
                    h=pixsi.hit.Hit(m[1]/16,m[0]-16,m[0])
                else:
                    h=pixsi.hit.Hit(m[1]/28,m[0]-28,m[0])
                rawHits.append(h)
        #Fit each block
        RecoveredSignals=[]
        for n,block in enumerate(M_blocks):
            result = pixsi.solver.solver_scipy(block,kernel)
            RecoveredSignals.append(result)
        #Create SP Hits
        for r,m in zip(RecoveredSignals,M_blocks):
            kl=len(kernel)
            Q=r[1:]
            t_st=r[0]
            t=m[0][0]+kl
            for n,q in enumerate(Q):
                h=pixsi.hit.Hit(q/(t-t_st),t_st,t)
                spHits.append(h)
                tmp=t_st
                t_st=t
                t=t+16 if n==0 else t+28
        #Save hits
        FinalHits.append((pn,[trueHits,rawHits,spHits]))
    
    for n,p in enumerate(FinalHits):
        if n!=1:
            continue
        print("Pixel ",p[0])
        print("True Hits: ",p[1][0])
        print("Raw Hits: ",p[1][1])
        print("SP Hits: ",p[1][2])
        dense_true = pixsi.util.make_dense_WF(p[1][0])
        dense_raw = pixsi.util.make_dense_WF(p[1][1])
        dense_sp = pixsi.util.make_dense_WF(p[1][2])
        print("Total Charge on Pixel")
        print("True: ",np.sum(dense_true))
        print("Raw: ",np.sum(dense_raw))
        print("SP: ",np.sum(dense_sp))
        time=np.linspace(0,160,1600)
        plt.plot(time,dense_true,label="True Signal")
        plt.plot(time,dense_raw,label="Raw Hits")
        plt.plot(time,dense_sp,label="SP Hits")
        plt.legend()
        plt.show()
    import pickle
    pickled_object = pickle.dumps(FinalHits)
    np.savez("FinalHits.npz", data=pickled_object)
    #loaded_data = np.load("my_object.npz")["data"]
    #loaded_object = pickle.loads(loaded_data)
  

@cli.command()
@click.option("-i","--input", type=str, required=False,
              help="Placeholder")
@click.pass_context
def eval(ctx,input):
    '''
    Evaluate Hits
    '''
    from pixsi.hit import Hit
    import numpy as np
    import pickle
    loaded_data = np.load(input)["data"]
    loaded_hits = pickle.loads(loaded_data)
    nonzerohits_true=[]
    nonzerohits_raw=[]
    nonzerohits_sp=[]
    
    totCharge_true=[]
    totCharge_raw=[]
    totCharge_sp=[]
    
    tStart_true=[]
    tStart_raw=[]
    tStart_sp=[]
    
    for p in loaded_hits:
        th=p[1][0]
        rh=p[1][1]
        sph=p[1][2]
        nonzerohits_true.append(np.sum([t.charge>1 for t in th]))
        nonzerohits_raw.append(np.sum([t.charge>1 for t in rh]))
        nonzerohits_sp.append(np.sum([t.charge>1 for t in sph]))
        
        totCharge_true.append(np.sum([t.charge*(t.end_time-t.start_time) for t in th]))
        totCharge_raw.append(np.sum([t.charge*(t.end_time-t.start_time) for t in rh]))
        totCharge_sp.append(np.sum([t.charge*(t.end_time-t.start_time) for t in sph]))
        
        tStart_true.append(th[0].start_time)
        tStart_raw.append(rh[0].start_time)
        tStart_sp.append(sph[0].start_time)
        
    x=np.linspace(0,len(loaded_hits),len(loaded_hits))
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    axs[0].plot(x, nonzerohits_true, label="TrueHits", color='b')
    axs[0].plot(x, nonzerohits_raw, label="RawHits", color='g')
    axs[0].plot(x, nonzerohits_sp, label="SPHits", color='r')
    axs[0].set_title("Number of Hits with Charge>1 per Pixel")
    axs[0].set_xlabel("Pixel")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(x, totCharge_true, label="TrueHits", color='b')
    axs[1].plot(x, totCharge_raw, label="RawHits", color='g')
    axs[1].plot(x, totCharge_sp, label="SPHits", color='r')
    axs[1].set_title("Total Charge per Pixel")
    axs[1].set_xlabel("Pixel")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(x, tStart_true, label="TrueHits", color='b')
    axs[2].plot(x, tStart_raw, label="RawHits", color='g')
    axs[2].plot(x, tStart_sp, label="SPHits", color='r')
    axs[2].set_title("Start Time of First Hit per Pixel")
    axs[2].set_xlabel("Pixel")
    axs[2].legend()
    axs[2].grid()


    plt.tight_layout()
    plt.show()
    
    
    for n,p in enumerate(loaded_hits):
        if n!=0:
            continue
        print("Pixel ",p[0])
        print("True Hits: ",p[1][0])
        print("Raw Hits: ",p[1][1])
        print("SP Hits: ",p[1][2])
        dense_true = pixsi.util.make_dense_WF(p[1][0])
        dense_raw = pixsi.util.make_dense_WF(p[1][1])
        dense_sp = pixsi.util.make_dense_WF(p[1][2])
        print("Total Charge on Pixel")
        print("True: ",np.sum(dense_true))
        print("Raw: ",np.sum(dense_raw))
        print("SP: ",np.sum(dense_sp))
        time=np.linspace(0,160,1600)
        plt.plot(time,dense_true,label="True Signal")
        plt.plot(time,dense_raw,label="Raw Hits")
        plt.plot(time,dense_sp,label="SP Hits")
        plt.legend()
        plt.show()
    
  
def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
