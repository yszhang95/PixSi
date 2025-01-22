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
@click.pass_context
def run(ctx,input):
    '''
    Runs Toy Simulation + Reconstruction 
    '''
    import numpy as np
    #define example true signal
    time_steps = np.linspace(0, 160, 1600)
    true_signal = np.zeros(len(time_steps))
    true_signal[63:62+5*9+1:5] = 1.0  # Example true signal
    true_signal[90:93] = 3.0
    true_signal[93:94] = 7.0
    true_signal[94:95] = 2.0# Example true signal
    #true_signal[187:187+4+1] = 1.0
    kernel=pixsi.toy_sim.kernel()
    #Get example induced Current
    current_response = pixsi.toy_sim.compute_current(true_signal, kernel, time_steps)
    
    
    trsh=200
    #Get example sparce triggered mesurements (time,value)
    M,q=pixsi.toy_sim.trigger(current_response,trsh)
    b=np.array(M)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, true_signal*150, label="True Signal (Charge)", linestyle="--")
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
    print(RecoveredSignals)
    Recovedred_dense=np.zeros(1600)
    t_st=int(RecoveredSignals[0][0])
    Q=RecoveredSignals[0][1:]
    dt1=M_blocks[0][0][0]+32-t_st
    dt2=M_blocks[0][1][0]-M_blocks[0][0][0]
    dt3=M_blocks[0][2][0]-M_blocks[0][1][0]
    Recovedred_dense[t_st:M_blocks[0][0][0]+32]=Q[0]/dt1
    Recovedred_dense[M_blocks[0][0][0]+32:M_blocks[0][1][0]+32]=Q[1]/dt2
    Recovedred_dense[M_blocks[0][1][0]+32:M_blocks[0][2][0]+32]=Q[2]/dt3
    print("Time Intervals: ",dt1,dt2,dt3)
    print("True signal=",np.sum(true_signal[:150]))
    print("Recovered signal=",np.sum(Recovedred_dense[:150]))
    t1=np.linspace(0,160,1600)
    plt.plot(t1,Recovedred_dense,label="Recovered Signal ")
    plt.plot(time_steps, true_signal*1, label="True Signal (Charge)", linestyle="--")
    
    modified_signal = np.zeros(len(true_signal))
    dt1_true=58+32-63
    dt2_true=16
    avg1=np.sum(true_signal[63:58+32])/dt1_true
    avg2=np.sum(true_signal[58+32:58+32+32])/dt2_true
    modified_signal[63:63+dt1_true]=avg1
    modified_signal[63+dt1_true:63+dt1_true+dt2_true]=avg2
    
    plt.plot(time_steps, modified_signal, label="True Signal (Modified)", linestyle="--")
    plt.legend()
    plt.plot()
    plt.xlim(6,11)
    plt.ylim(0,1.5)
    plt.show()


def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
