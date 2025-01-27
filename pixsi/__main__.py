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
    plt.plot(time_steps,kernel)
    plt.plot(time_steps,np.cumsum(kernel))
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
        kernel=pixsi.toy_sim.getKernel(kernelresp)
    import matplotlib.pyplot as plt
    import numpy as np
    #define example true signal
    track1 = pixsi.toy_sim.sim_MIP(0,0,10,10)
    
    

def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
