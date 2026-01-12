import sys
import json
import click
import pixsi
import torch
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

    #meas , true_charges = pixsi.util.extract_TRED_by_tpc(input)
    meas , true_charges, true_wfs = pixsi.util.extract_TRED_test(input)
    import sys
    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)

    tpc=1
    
    #print("True Hits: ",true_charges[tpc])
    
    print("Measurements : ",meas)
    
    ext_meas = pixsi.preproc.extend_measurements(meas,5000,0.05)
    
    print("Extended Measurements: ",ext_meas)
    
    signals = pixsi.preproc.define_signals_simple(ext_meas,5000,0.05)
    print("Defined Signals: ",signals)
    print("# of TPCs: ",len(meas))
    print("# of measurements: ",len(meas[tpc]))
    print("# of True Charges: ",len(true_charges[tpc]))
    #print("# of signals: ",len(signals))
    response=pixsi.kernels.getKernel_NDLar(kernelresp)
    
    import matplotlib.pyplot as plt
    plt.plot(np.cumsum(response[0][0]),label='middle pixel')
    plt.plot(np.cumsum(response[0][1]),label='+1 right')
    plt.plot(np.cumsum(response[1][0]),label='+1 top')
    plt.plot(np.cumsum(response[0][2]),label='+2 right')
    plt.plot(np.cumsum(response[2][0]),label='+2 top')
    plt.plot(np.cumsum(response[1][1]),label='+1.5')
    plt.plot(np.cumsum(response[2][2]),label='+2.5')
    #plt.xlim(1111,1319)
    plt.legend()
    plt.show()
    
    
    m=meas#[tpc]
    pixelidx = [item[0][0] for item in m]
    pixelidy = [item[0][1] for item in m]
    time = [item[1] for item in m]
    charge = [item[2] for item in m]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pixelidx, pixelidy, time, c=charge, cmap='viridis')
    cb = plt.colorbar(sc, ax=ax, label='Charge')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.set_zlabel('Time')
    plt.title('3D Scatter of Pixels vs Time Colored by Charge')

    plt.show()
    import time
    t0 = time.time()
    
    #response=[[response[0][0]]]
    sp_result, pixel_block_param_map = pixsi.solver_2D_fast_simple.solver_2D_scipy_simple(ext_meas,signals,response)

    t1 = time.time()
    #print("SP_results: ", sp_result)

    print("Minimization Took Arrpoximately : ",(t1-t0)/60.0," min")
    raw_hits,sp_hits,true_hits,true_hit_perpix,eff_hits=pixsi.util.create_hits(meas, sp_result, true_charges,tpc,0,response[0][0],time_tick=0.05)
    FinalHits=[raw_hits,sp_hits,true_hits,true_hit_perpix,eff_hits]
    import pickle
    pickled_object = pickle.dumps(FinalHits)
    #print("WE ARE NOT SAVING ANYTHING")
    np.savez("FinalHits_tred_nogrid_complex_noise_3p5k.npz", data=pickled_object)
    


@cli.command()
@click.option("-i","--input", type=str, required=False,
              help="Placeholder")
@click.option("-k","--kernelresp",type=click.Path(),required=False,
              help="Path to Field Response")
@click.pass_context
def run_SP_tred_burst(ctx,input,kernelresp):
    '''
        Run Signal Processing on TRED output with simulated burst mode
    '''

    meas , true_charges, true_wfs = pixsi.toy_sim.burstMode(input)
    
    
    burst_meas = pixsi.util.build_burst_measurements(true_wfs)
    
    import sys
    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)
    ext_meas = pixsi.preproc.extend_measurements(meas,5000,0.05)
    #print("True Meas : ",true_charges)
    #print("True WF : ",true_wfs)
    #print("Burst Meas : ",burst_meas)

    response=pixsi.kernels.getKernel_NDLar(kernelresp,kind="cumulative")

    

    
 #   plt.scatter(nptoplot[:,0],nptoplot[:,1],label="pixel = (143,131)")
 #   plt.scatter(nptoplot2[:,0],nptoplot2[:,1],label="pixel = (143,133)")
 #   plt.legend(loc='upper left')
 #   plt.show()
    
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
       
    
    ker = pixsi.deconv3d.Kernel3D(response)

    engine = pixsi.deconv3d.Deco3D(ker)
    
    
    # Choose if you want to do sand alone tests with toy simulation or run deconvolution on TRED/Data input
    dotests=False
    
    run_data=True
    
    
    if dotests:
        pixsi.deconv_visual_checks.run_demo(
            ker, t0=380, threshold=5,
            ramp_pre=None, lam0=1e-2, lam_hf=180, lam_exp=3.0, taper_frac=0.1,align="model", align_fractional=False
        )


        pixsi.deconv_visual_checks.run_perfect_reco_check(
            ker, t0=380, sigma=4.0, total_charge=100.0,
            lam0=0, taper_frac=0.0, use_gaussian=True, gauss_sigma_frac=500
        )
    
    
        out = pixsi.deconv3d_demo.run_demo(
        ker,                                   # your provided kernel
        T=12000, Nx=13, Ny=13,
        pos_a=(7,7), t0_a=1714, sigma_a=14.0, Q_a=320.0,
        pos_b=(7,8), t0_b=1690, sigma_b=4.0, Q_b=10.0,
        #threshold=0.1, nsamples=25, spacing=16,
        threshold=0, nsamples=2000, spacing=16,
        lam0=0.0, lam_hf=155, lam_exp=3.0,
        taper_frac_t=0.0,
        deconv_domain="dy",
        align="model", align_fractional=True,
        clamp_nonneg=False,
        show=True,
        )

    def shift_1d(arr, k):
        out = np.zeros_like(arr)

        if k > 0:
            # shift right
            out[k:] = arr[:-k]
        elif k < 0:
            # shift left
            k = -k
            out[:-k] = arr[k:]
        else:
            return arr.copy()

        return out
    #for n,m in enumerate(burst_meas):
        #if m[0]==(75,129):
                #burst_meas[n]=(m[0],m[1],m[2])

    if run_data:
        out = engine.run(
            measurements=burst_meas,
            measurement_type="cumulative",
            true_list=true_charges, true_is_incremental=True,
            roi_margin=0,
            lam0=0, lam_hf=2, lam_exp=3.0,
            align="NONEmodel", align_fractional=False,
            # Stage B
            refine_nonneg=False,
            refine_iters=10,        # start with 5–20
            refine_step=0.05,       # tune smaller if it oscillates
            refine_lam=1e-2,        # small L2 penalty on q
            refine_verbose=True,
            # plotting
            pixels_to_plot=[(342,109),(343,109)],
            show_maps=True,
            show=True
        )
#    dt_map = None
#    dt=None
#
#    pixsi.deconv_sanity_checks.run_all(ker,cfg)


#    out = engine.run(meas, cumulative=True)

#    q_hat = out["q"]               # Tensor [Nt_out, Nx_roi, Ny_roi] on `device`
#    roi   = out["roi"]             # dict with origins: {"t0": ..., "x0": ..., "y0": ...}
#    meta  = out["meta"]

#    print("Deconvolution complete.")
#    print("q_hat shape (Nt, Nx, Ny):", tuple(q_hat.shape))
#    print("ROI origin (indices in your input coordinate system):", roi)
#    print("Meta:", meta)
   

#    _ = pixsi.analyze_after_deconv.analyze_and_plot(
#    out,                        # dict returned by Deconv3D.run(...)
#    measurements=meas,     # (x,y,t,q) or ((x,y),t,q)
#    true_list=true_charges,        # same shape; per-interval truth if you have it
#    response_quadrant=response,# [5][5][L] cumulative response (center not used here)
#    dt=None,                    # set if your t are floats
     # try the defaults first
#    xy_mode_meas="xy",     # if you still see the left-edge yellow stripe, try "yx"
#    xy_mode_true="xy",     # independently toggle this if your truth was built with swapped axes

#    do_spatial_align_xy=True,   # will print best (dx,dy) and apply it for plots
#    xy_align_max_shift=6,

#    meas_are_cumulative=False,
#    scale_meas_to="true",
#    scale_deconv_to="true",
#    do_time_align=True,
#)
   
    #raw_hits,sp_hits,true_hits,true_hit_perpix,eff_hits=pixsi.util.create_hits(meas, sp_result, true_charges,tpc,0,response[0][0],time_tick=0.05)
    #FinalHits=[raw_hits,sp_hits,true_hits,true_hit_perpix,eff_hits]
    #import pickle
    #pickled_object = pickle.dumps(FinalHits)
    #print("WE ARE NOT SAVING ANYTHING")
    #np.savez("FinalHits_tred_nogrid_complex_noise_3p5k.npz", data=pickled_object)





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
        #plt.plot(time,dense_true,label="True Signal")
        plt.plot(time,dense_raw,label="Raw Hits")
        plt.plot(time,dense_sp,label="SP Hits")
        plt.legend()
        plt.show()
    
 
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
        

    
    print("Total Cllected Raw Charge = ",np.sum(charge_per_hit_raw))
    print("Total Cllected SP Charge = ",np.sum(charge_per_hit_sp))
    print("Total Cllected TRUE Charge = ",np.sum(charge_per_hit_true))
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(0,len(charge_per_hit_true),len(charge_per_hit_true))
    plt.plot(x,charge_per_hit_true,label="True Hits")
    plt.plot(x,charge_per_hit_raw,label="Raw Hits")
    plt.plot(x,charge_per_hit_sp,label="SP Hits")
    plt.xlabel("hit_ID")
    plt.ylabel("Charge")
    plt.legend()
    plt.show()
    
    raw_hits_pixid={}
    
    for i in loaded_hits[0]:
        if i.pixel_ID in raw_hits_pixid:
            raw_hits_pixid[i.pixel_ID].append(i.charge*(i.end_time-i.start_time))
        else:
            raw_hits_pixid[i.pixel_ID]=[i.charge*(i.end_time-i.start_time)]
            
    sp_hits_pixid={}
    
    for i in loaded_hits[1]:
        if i.pixel_ID in sp_hits_pixid:
            sp_hits_pixid[i.pixel_ID].append(i.charge*(i.end_time-i.start_time))
        else:
            sp_hits_pixid[i.pixel_ID]=[i.charge*(i.end_time-i.start_time)]
            
    true_hits_pixid={}
    
    for i in loaded_hits[2]:
        if i.pixel_ID in true_hits_pixid:
            true_hits_pixid[i.pixel_ID].append(i.charge*(i.end_time-i.start_time))
        else:
            true_hits_pixid[i.pixel_ID]=[i.charge*(i.end_time-i.start_time)]
    
    raw_charge_per_pixel=[]
    sp_charge_per_pixel=[]
    true_charge_per_pixel=[]
    
    fake_true_hits = loaded_hits[3]
    #print(fake_true_hits)
    considered_pixels = []
    for key in raw_hits_pixid:
        considered_pixels.append((key[0],key[1]))
    sorted(considered_pixels,key=lambda x:x[0])
    raw_hits_per_pix = []
    sp_hits_per_pix = []
    for key in considered_pixels:
        raw_hits_per_pix.append(len(raw_hits_pixid[key]))
        sp_hits_per_pix.append(len(sp_hits_pixid[key]))
        raw_charge_per_pixel.append(sum(raw_hits_pixid[key]))
        sp_charge_per_pixel.append(sum(sp_hits_pixid[key]))
        true_charge_per_pixel.append(sum([i.charge for i in fake_true_hits if key==i.pixel_ID]))
    rhcp = np.array(raw_charge_per_pixel)
    sphcp = np.array(sp_charge_per_pixel)
    thcp = np.array(true_charge_per_pixel)
        
    mean_raw = np.mean((thcp-rhcp)/thcp)
    std_raw = np.std((thcp-rhcp)/thcp)

    mean_sp = np.mean((thcp-sphcp)/thcp)
    std_sp = np.std((thcp-sphcp)/thcp)

    plt.hist((rhcp-thcp)/thcp,bins=100,range=(-1,1), alpha=0.7,label=f"Raw : $\mu$={mean_raw:.3f}, $\sigma$={std_raw:.3f}")
    plt.hist((sphcp-thcp)/thcp,bins=100,range=(-1,1), alpha=0.7,label=f"SP : $\mu$={mean_sp:.3f}, $\sigma$={std_sp:.3f}")
    plt.title("Charge Per Pixel")
    plt.ylabel("# of Pixels")
    plt.xlabel("Res.=(X-True)/True, %")
    plt.legend()
    plt.show()
    
    
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    pixels = range(len(thcp))
    # Top plot: Charges
    ax1.plot(pixels,rhcp, label="Measured Charge")
    ax1.plot(pixels,sphcp, label="SP Charge")
    ax1.plot(pixels,thcp, label="True Charge")
    ax1.set_title("Charge Per Pixel")
    ax1.set_ylabel("Charge")
    ax1.legend()
    ax1.grid(True)

    # Bottom plot: Residuals as points
    residual_raw = 100*(rhcp - thcp)/thcp
    residual_sp = 100*(sphcp - thcp)/thcp
    
    ax2.scatter(pixels, residual_raw, label="Measured", alpha=0.6, s=10)
    ax2.scatter(pixels, residual_sp, label="SP", alpha=0.6, s=10)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel("Res.=(X-True)/True, %")
    ax2.set_xlabel("Pixel # sorted in increasing order along Y axis")
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(sp_hits_per_pix, label="Hits per Pixel", marker='x', linestyle='', alpha=0.6)
    ax3.set_ylabel("Hits/Pixel")
    ax3.set_xlabel("Pixel")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(8, 5))
    plt.scatter(raw_hits_per_pix, residual_raw, alpha=0.6, s=15)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Raw Hits per Pixel")
    plt.ylabel("Res.=(Measured - True)/True, %")
    plt.title("Residual vs Raw Hits per Pixel")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for p in range(3):
        print("New Call")
        hit_raw = np.zeros(12000)
        hit_sp = np.zeros(12000)
        hit_true = np.zeros(12000)
        for hit_ID in raw_hits:
            th=true_hits[hit_ID]
            rh=raw_hits[hit_ID]
            sph=sp_hits[hit_ID]
            pixel = [i for i in unique_pixels][p]
            if th.pixel_ID==pixel:
                print("True Charge : ",th.charge)
                hit_true[th.start_time:th.end_time+1]=th.charge
            if rh.pixel_ID==pixel:
                print("Raw Charge : ",rh.charge)
                hit_raw[rh.start_time:rh.end_time+1]=rh.charge
            if sph.pixel_ID==pixel:
                print("SP Charge : ",sph.charge)
                hit_sp[sph.start_time:sph.end_time+1]=sph.charge
                print("Considered Hit: ", sph.pixel_ID)
        plt.plot(hit_true,label="True Hits")
        plt.plot(hit_raw,label="Raw Hits")
        plt.plot(hit_sp,label="SP Hits")
        plt.xlabel("Time in ticks")
        plt.ylabel("Charge")
        plt.xlim(0,4000)
        plt.legend()
        #plt.show()
    

@cli.command()
@click.option("-i","--input", type=str, required=False,
              help="Input Hits")
@click.pass_context
def eval_tred_true(ctx,input):
    '''
    Evaluate Hits created from TRED
    '''
    from pixsi.hit import Hit
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    loaded_data = np.load(input)["data"]
    loaded_hits = pickle.loads(loaded_data)

    chargeunits=1
    raw_hits = {hit.hit_ID: hit for hit in loaded_hits[0]}
    sp_hits = {hit.hit_ID: hit for hit in loaded_hits[1]}
    true_hits = {hit.hit_ID: hit for hit in loaded_hits[2]}
    
    raw_hits_per_pix = {}
    sp_hits_per_pix = {}
    true_hits_per_pix = {}
    true_hits_per_pix_tot = {}
    eff_hits_per_pix = {}

    for h in loaded_hits[0]:
        if h.pixel_ID in raw_hits_per_pix:
            raw_hits_per_pix[h.pixel_ID].append(h)
        else:
            raw_hits_per_pix[h.pixel_ID]=[h]
    for h in loaded_hits[1]:
        if h.pixel_ID in sp_hits_per_pix:
            sp_hits_per_pix[h.pixel_ID].append(h)
        else:
            sp_hits_per_pix[h.pixel_ID]=[h]
    for h in loaded_hits[2]:
        if h.pixel_ID in true_hits_per_pix:
            true_hits_per_pix[h.pixel_ID].append(h)
            true_hits_per_pix_tot[h.pixel_ID]+=h.charge/chargeunits
        else:
            true_hits_per_pix[h.pixel_ID]=[h]
            true_hits_per_pix_tot[h.pixel_ID]=h.charge/chargeunits
    for h in loaded_hits[4]:
        if h.pixel_ID in eff_hits_per_pix:
            eff_hits_per_pix[h.pixel_ID].append(h)
        else:
            eff_hits_per_pix[h.pixel_ID]=[h]


    print("Number of raw_hits = ",len(raw_hits))
    print("Number of sp_hits = ",len(sp_hits))
    print("Number of true_hits = ",len(true_hits))
    
    count=0
    pixels=[]
    rch=[]
    spch=[]
    tch=[]

    for p,rhs in raw_hits_per_pix.items():
        #if count>16:
        #    break
        if p!=(197, 74):
            continue
        pixels.append(pixels)
        count+=1
        rwf = np.zeros(12000)
        rawcharge=0
        spcharge=0
        tcharge=0
        for i in rhs:
            rwf[i.start_time:i.end_time]=i.charge #* (i.end_time-i.start_time)
            rawcharge+=i.charge * (i.end_time-i.start_time)
        spwf = np.zeros(12000)
        for i in sp_hits_per_pix[p]:
            spwf[i.start_time:i.end_time]=i.charge #* (i.end_time-i.start_time)
            spcharge+=i.charge * (i.end_time-i.start_time)
        twf = np.zeros(12000)
        if p not in true_hits_per_pix: continue
        for i in true_hits_per_pix[p]:
            twf[i.start_time:i.end_time]=i.charge/chargeunits
            tcharge+=i.charge*(i.end_time-i.start_time)/chargeunits
        effwf = np.zeros(12000)
        
        effcharge=0
        if p not in eff_hits_per_pix: continue
        for i in eff_hits_per_pix[p]:
            effwf[i.start_time]=i.charge/chargeunits
            effcharge+=i.charge/chargeunits
        rch.append(rawcharge)
        spch.append(spcharge)
        tch.append(tcharge)
        #if abs((rawcharge-tcharge)/tcharge)>0.005:
            #print(count)
        if count<3: #count == 1 or count==4 or count == 15:
            plt.title(f"Pixel # {p}")
            plt.plot(rwf,label=f"Raw:{rawcharge:.02f}")
            plt.plot(spwf,label=f"SP:{spcharge:.02f}")
            plt.plot(twf,label=f"True:{tcharge:.02f}")
            plt.plot(effwf,label=f"Eff:{effcharge:.02f}")
            plt.legend(loc="upper right")
            #plt.xlim(300,1900)
            plt.show()
    npraw = np.array(rch)
    npsp = np.array(spch)
    npt = np.array(tch)
    res_r = (npraw-npt)/npt
    res_sp = (npsp-npt)/npt
    plt.hist(100*res_r,bins=100,range=(-2500,2500),alpha=0.7,label=f"Raw Res, $\mu=${np.median(100*res_r):.02f}, $\sigma=${np.std(100*res_r):.02f}")
    plt.hist(100*res_sp,bins=100,range=(-2500,2500),alpha=0.7,label=f"SP Res, $\mu=${np.median(100*res_sp):.02f}, $\sigma=${np.std(100*res_sp):.02f}")
    plt.legend(loc='upper right')
    plt.title("Per Pixel")
    plt.show()
    
    
    res_hit_r = []
    res_hit_sp = []
    true_left = []
    dif1 = []
    dif2 = []
    allhits = []
    fractionhits = []
    for k,v in raw_hits.items():
        v_sp = sp_hits[k]
        if k not in true_hits:
            print("No True Info")
            print("Reco Charge : ",v.charge ,"; SP Charge : ", v_sp.charge)
            dif1.append(100*(v.charge-v_sp.charge)/v.charge)
            continue
        v_t = true_hits[k]
        if 100*chargeunits*(v.charge - v_t.charge/chargeunits)/v_t.charge > 500:
            print("With True and Large difference > 500%")
            print("Reco Charge : ",v.charge ,"; SP Charge : ", v_sp.charge,"; True Charge : ",v_t.charge)
            if (v.charge-v_sp.charge)/v.charge < -0.1:
                fractionhits.append((v.pixel_ID[0],v.pixel_ID[1],v.start_time))
            dif2.append(100*(v.charge-v_sp.charge)/v.charge)
        else:
            allhits.append((v.pixel_ID[0],v.pixel_ID[1],v.start_time))

            res_hit_r.append(100*chargeunits*(v.charge - v_t.charge/chargeunits)/v_t.charge)
            res_hit_sp.append(100*chargeunits*(v_sp.charge-v_t.charge/chargeunits)/v_t.charge)
    for k,v in true_hits.items():
        if k not in raw_hits and v.pixel_ID in raw_hits_per_pix:
            true_left.append(v.charge/1000/true_hits_per_pix_tot[v.pixel_ID])
            
    
    x_hits, y_hits, z_hits = zip(*allhits) if allhits else ([], [], [])
    x_out, y_out, z_out = zip(*fractionhits) if fractionhits else ([], [], [])

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plots
    ax.scatter(x_hits, y_hits, z_hits, c='blue', label="All Hits", alpha=0.16)
    ax.scatter(x_out, y_out, z_out, c='red', label="Outliers", alpha=0.8)

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("3D Scatter Plot of Hits and Outliers")

    plt.show()
    
    
    
    plt.hist(res_hit_r,bins=100,range=(-100,200),alpha=0.7,label=f"Raw Res, $\mu=${np.median(res_hit_r):.02f}, $\sigma=${np.std(res_hit_r):.02f}")
    plt.hist(res_hit_sp,bins=100,range=(-100,200),alpha=0.7,label=f"SP Res, $\mu=${np.median(res_hit_sp):.02f}, $\sigma=${np.std(res_hit_sp):.02f}")
    plt.legend(loc='upper right')
    plt.title("Per Hit, raw-true/true <500%")
    plt.show()
    
    d1 = np.array(dif1)
    d2 = np.array(dif2)
    
    plt.hist(d1,bins=50,range=(-100,100),alpha=0.7,label=f"$\mu=${np.mean(d1):.02f}")
    plt.legend(loc='upper right')
    plt.title("Raw-SP / Raw , %  in hits without True charge")
    plt.show()

    plt.hist(d2,bins=50,range=(-100,100),alpha=0.7,label=f"$\mu=${np.mean(d2):.02f}")
    plt.legend(loc='upper right')
    plt.title("Raw-SP / Raw , %  in hits with extremely law True charge")
    plt.show()

    plt.hist(true_left,bins=100,range=(-1,1),alpha=0.7,label=f"True left, $\mu=${np.mean(true_left):.02f}, $\sigma=${np.std(true_left):.02f}")
    plt.title("Per Hit Left")
    plt.legend(loc='upper right')
    plt.show()
            
    

def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
