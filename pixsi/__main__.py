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
    Toy Sim + SP for a single pixel
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
    Toy Sim + SP for two MIP tracks [NO long-range ind.]
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
    track2 = pixsi.toy_sim.sim_MIP(190,20,80,10)
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
        FinalHits.append([pn,[trueHits,rawHits,spHits]])
    
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
@click.option("-k","--kernelresp",type=click.Path(),required=False,
              help="Path to Field Response")
@click.pass_context
def run_2D_full(ctx,input,kernelresp):
    '''
        Toy Sim + SP for two MIP tracks with +-1 pixle for long range ind.
    '''
    if not kernelresp:
        print("Path to FR was not provided. Using Toy Kernel")
        kernel=pixsi.toy_sim.kernel()
        kernel_ind=pixsi.toy_sim.kernel()
    else:
        kernel=pixsi.toy_sim.getKernel(kernelresp)*100
        kernel_ind=pixsi.toy_sim.getKernel_Ind(kernelresp)*100
    kernel=kernel[kernel!=0]
    kernel_ind=kernel_ind[kernel_ind>0]
    import matplotlib.pyplot as plt
    import numpy as np
    #plt.plot(np.cumsum(kernel),label="kernel middle")
    #plt.plot(np.cumsum(kernel_ind),label="kernel ind")
    #plt.legend()
    #plt.show()
    #define example true signal
    track1 = pixsi.toy_sim.sim_MIP(20,0,300,45)
    track2 = pixsi.toy_sim.sim_MIP(30,0,300,10)
    print("Total Track1 Charge: ",np.sum(np.array([np.sum(i) for i in track1])))
    if len(track2)>0: print("Total Track2 Charge: ",np.sum(np.array([np.sum(i) for i in track2])))
    if len(track2)>0 and len(track1)<len(track2):
        arr = [np.zeros(1600) for _ in range(len(track2)-len(track1))]
        track1=track1+arr
    pixels = np.array([a + b for a, b in zip(track1, track2)]) if len(track2)>0 else np.array(track1)
    #np.insert(pixels,0,np.zeros(1600))
    #np.append(pixels,np.zeros(1600))
    #pixels=[pixels[0]]
    print("Used Pixels: ",len(pixels))
    meas,blocks,hits_true_raw = pixsi.toy_sim.simActivity_toy(pixels,kernel,kernel_ind)
    
    #print("Measurements : ",meas)
    
    ext_meas = pixsi.preproc.extend_measurements(meas,5000) # only shifts measurements by 1.6 but has an ability to add threshold measurement upfront and possibly zero measurement at the end of the sequence
    
    #print("Extended Measurements: ",ext_meas)
    
    #signals = pixsi.preproc.define_signals(ext_meas,len(kernel),5000)
    signals = pixsi.preproc.define_signals_simple(ext_meas,len(kernel),5000)
    #print("Defined Signals: ",signals)
    print("# of measurements: ",len(meas))
    print("# of signals: ",len(signals))
    response=[[kernel, kernel_ind], [kernel_ind, kernel_ind]]
    
    from .util import uniform_charge_cum_current_part as current_part

        
    import time
    t0 = time.time()
    
    
    sp_result, pixel_block_param_map = pixsi.solver_2D_fast_simple.solver_2D_scipy_simple(ext_meas,signals,response)
    
    #sp_result, pixel_block_param_map = pixsi.solver_2D.solver_2D_scipy(blocks,kernel,kernel_ind)
    
    t1 = time.time()

    print("Minimization Took Arrpoximately : ",(t1-t0)/60.0," min")
    #print(hits_true_raw)
    #print("SP Result: ", sp_result)
    FinalHits = []
    idx_param=0
    for npi,p in enumerate(hits_true_raw):
        spHits=[]
        for s in sp_result:
            if s[1]!=npi+1:
                continue
            spHits.append(pixsi.hit.Hit(s[2]/s[4],s[3],s[3]+s[4]))
        hits_true_raw[npi][1].append(spHits)
        FinalHits.append([npi,hits_true_raw[npi][1]])
    import pickle
    pickled_object = pickle.dumps(FinalHits)
    np.savez("FinalHits_2d_test_long_fast.npz", data=pickled_object)
    


@cli.command()
@click.option("-i","--input", type=str, required=False,
              help="Placeholder")
@click.option("-k","--kernelresp",type=click.Path(),required=False,
              help="Path to Field Response")
@click.pass_context
def run_SP_tred(ctx,input,kernelresp):
    '''
        Run Signal Processing on TRED output
    '''
    kernel=pixsi.toy_sim.getKernel(kernelresp,0.05,0.05)*100
    kernel_ind=pixsi.toy_sim.getKernel_Ind(kernelresp,0.05,0.05)*100
    
    kernel=kernel[kernel!=0]
    kernel_ind=kernel_ind[kernel_ind>0]
    
    #import matplotlib.pyplot as plt
    #plt.plot(kernel)
    #plt.plot(kernel_ind)
    #plt.show()

    meas , true_charges = pixsi.util.extract_TRED_by_tpc(input)
    import sys
    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)

    tpc=1
    
    
    
    print("Measurements : ",meas[tpc][:5])
    
    ext_meas = pixsi.preproc.extend_measurements(meas[1],5000,0.05)
    
    print("Extended Measurements: ",ext_meas[:5])
    
    signals = pixsi.preproc.define_signals_simple(ext_meas,len(kernel),5000,0.05)
    print("Defined Signals: ",signals[:5])
    print("# of TPCs: ",len(meas))
    print("# of measurements: ",len(meas[tpc]))
    print("# of True Charges: ",len(true_charges[tpc]))
    #print("# of signals: ",len(signals))
    response=[[kernel, kernel_ind,kernel_ind / 10, kernel_ind / 100], [kernel_ind/10, kernel_ind/20,kernel_ind/30, kernel_ind/40],[kernel_ind/100, kernel_ind/120,kernel_ind/130, kernel_ind/140]]
    

        
    import time
    t0 = time.time()
    
    
    sp_result, pixel_block_param_map = pixsi.solver_2D_fast_simple.solver_2D_scipy_simple(ext_meas,signals,response)

    t1 = time.time()

    print("Minimization Took Arrpoximately : ",(t1-t0)/60.0," min")
    raw_hits,sp_hits,true_hits=pixsi.util.create_hits(meas[tpc], sp_result, true_charges[tpc],tpc,0,time_tick=0.05)
    FinalHits=[raw_hits,sp_hits,true_hits]
    import pickle
    pickled_object = pickle.dumps(FinalHits)
    np.savez("FinalHits_tred.npz", data=pickled_object)
    






@cli.command()
@click.option("-i","--input", type=str, required=False,
              help="Input Hits")
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
    adj=0
    for npp,p in enumerate(loaded_hits):
        th=p[1][0]
        rh=p[1][1]
        sph=p[1][2]
        if len(sph)==0:
            adj+=1
            continue
        print(th)
        if len(th)>0:
            nonzerohits_true.append(np.sum([t.charge>1 for t in th]))
            totCharge_true.append(np.sum([t.charge*(t.end_time-t.start_time) for t in th]))
            tStart_true.append(th[0].start_time)
        else:
            nonzerohits_true.append(0)
            totCharge_true.append(0)
            tStart_true.append(0)
        
        nonzerohits_raw.append(np.sum([t.charge>1 for t in rh]))
        nonzerohits_sp.append(np.sum([t.charge>1 for t in sph]))
        
        totCharge_raw.append(np.sum([t.charge*(t.end_time-t.start_time) for t in rh]))
        totCharge_sp.append(np.sum([t.charge*(t.end_time-t.start_time) for t in sph]))
        
        tStart_raw.append(rh[0].start_time)
        tStart_sp.append(sph[0].start_time)
        
    x=np.linspace(0,len(loaded_hits)-adj,len(loaded_hits)-adj)
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axs[0].plot(x, nonzerohits_true, label="TrueHits", color='b')
    axs[0].plot(x, nonzerohits_raw, label="RawHits", color='g')
    axs[0].plot(x+adj, nonzerohits_sp, label="SPHits", color='r')
    axs[0].set_title("Number of Hits with Charge>1 per Pixel")
    axs[0].set_xlabel("Pixel")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(x[2:], totCharge_true[2:], label="TrueHits", color='b')
    axs[1].plot(x[2:], totCharge_raw[2:], label="RawHits", color='g')
    axs[1].plot(x[2:], totCharge_sp[1:-1], label="SPHits", color='r')
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
    tru_np = np.array(totCharge_true[2:])
    raw_np = np.array(totCharge_raw[2:])
    sp_np = np.array(totCharge_sp[1:-1])
    plt.hist((tru_np[tru_np>0]-raw_np[raw_np>0])/tru_np[tru_np>0],label="Raw Meas")
    plt.hist((tru_np[tru_np>0]-sp_np[sp_np>0])/tru_np[tru_np>0],label="SP Meas")
    plt.legend()
        
    plt.xlabel('$(Charge_{True}-Charge_{meas})/Charge_{True}$')
    plt.ylabel('# of Pixels')
    plt.title('Observed Charge Per Pixel')
    plt.show()
    for n,p in enumerate(loaded_hits):
        if n>7:
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
    #    plt.show()
    
 
@cli.command()
@click.option("-i","--input", type=str, required=False,
              help="Input Hits")
@click.pass_context
def eval_tred(ctx,input):
    '''
    Evaluate Hits created from TRED
    '''
    from pixsi.hit import Hit
    import numpy as np
    import pickle
    loaded_data = np.load(input)["data"]
    loaded_hits = pickle.loads(loaded_data)

    
    raw_hits = {hit.hit_ID: hit for hit in loaded_hits[0]}
    sp_hits = {hit.hit_ID: hit for hit in loaded_hits[1]}
    true_hits = {hit.hit_ID: hit for hit in loaded_hits[2]}
    
    print("Number of raw_hits = ",len(raw_hits))
    print("Number of sp_hits = ",len(sp_hits))
    print("Number of true_hits = ",len(true_hits))
    
    charge_per_hit_raw = []
    charge_per_hit_sp = []
    charge_per_hit_true = []
    
    assert len(raw_hits) == len(sp_hits) == len(true_hits), "Hit arrays are not of equal length"
    unique_pixels=set()
    for hit_ID in raw_hits:
        th=true_hits[hit_ID]
        rh=raw_hits[hit_ID]
        sph=sp_hits[hit_ID]
        unique_pixels.add(rh.pixel_ID)
        charge_per_hit_raw.append(rh.charge)
        charge_per_hit_sp.append(sph.charge)
        charge_per_hit_true.append(th.charge)
        
    hit_raw = np.zeros(12000)
    hit_sp = np.zeros(12000)
    hit_true = np.zeros(12000)
    for hit_ID in raw_hits:
        th=true_hits[hit_ID]
        rh=raw_hits[hit_ID]
        sph=sp_hits[hit_ID]
        pixel = next(iter(unique_pixels))
        if th.pixel_ID==pixel:
            hit_true[th.start_time:th.end_time+1]=th.charge
        if rh.pixel_ID==pixel:
            hit_raw[rh.start_time:rh.end_time+1]=rh.charge
        if sph.pixel_ID==pixel:
            hit_sp[sph.start_time:sph.end_time+1]=sph.charge
            
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(charge_per_hit_true,label="True Hits")
    plt.plot(charge_per_hit_raw,label="Raw Hits")
    plt.plot(charge_per_hit_sp,label="SP Hits")
    plt.xlabel("hit_ID")
    plt.ylabel("Charge")
    plt.legend()
    plt.show()
    
    plt.plot(hit_true,label="True Hits")
    plt.plot(hit_raw,label="Raw Hits")
    plt.plot(hit_sp,label="SP Hits")
    plt.xlabel("Time in ticks")
    plt.ylabel("Charge")
    plt.legend()
    plt.show()
    

def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
