import pickle
import sys
from neuron import h
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../SCSinSCIBiophysicalModel')
from tools.general_tools import ensure_dir
import cells as cll
import tools.neuron_functions as nf
import tools.plotting_tools as pt

def record_mn_membrane_potential_supraspinal_scs_simulation(scs_amp: float,
                                   scs_freq: int, 
                                   perc_supra_intact: float = 1, 
                                   supra_inhibit: bool = False, 
                                   save_data_folder: str = '', 
                                   plot_sim: bool = False):
    """ 
    Record motoneuron membrane potential with combinations of modulation from 
    SCS and residual supraspinal inputs, as is done in Figure 2 (Balaguer et al., 2025).

    Simulation starts with 250ms of supraspinal inputs alone, then 
    250ms of SCS input alone, then 250ms of SCS and supraspinal inputs (total time = 750ms)

    Args: 
        - scs_amp: float, amplitude of stimulation (% of max # of SCS inputs)
        - scs_freq: int, frequency of stimulation (Hz)
        - perc_supra_intact: float, remaining supraspinal inputs (% of intact model = 300)
        - supra_inhibit: bool, determines if supraspinal inputs are inhibitory (True = inhibitory supraspinal inputs)
        - simulation_duration: int, length of simulation (ms)
        - save_data_folder: str, name of folder to save data (if not provided, data will not be saved)
        - plot_sim: bool, plot simulation results
    
    Out: N/A, see `save_data_folder` and `plot_sim` for simulation output options
    """

    h.load_file('stdrun.hoc')
    np.random.seed(672945) # set the seed so the network is always the same

    # Set SCS parameters
    num_scs_total = 60
    num_scs_effective = int(scs_amp*num_scs_total)

    # Set supraspinal parameters
    num_supraspinal_total = 300
    num_supraspinal = int(num_supraspinal_total*perc_supra_intact)
    rate_supraspinal = 60 # firing rate (in Hz)
    
    # Set MN parameters
    num_mn = 10
    mn_drug = True
    mn_avg_diameter = 36
    
    # Set synaptic parameters
    synaptic_weight_scs = 0.000148
    synaptic_weight_supra = 0.000148
    if supra_inhibit: 
        synaptic_weight_supra = synaptic_weight_supra*5 # inhibitory synapses are 5x stronger
    shape = 1.2
    tau = 2

    # Set timing of input starts/stops
    supraspinal_start = 0
    scs_start = 250
    supraspinal_restart = 500
    simulation_duration = 750
    
    # Create lists to hold the neurons and spike recorders
    supraspinal_neurons = []
    scs_neurons = []
    supraspinal_spike_times = []
    scs_pulse_times = []

    # Create a population of SCS pulses and record their spikes
    if num_scs_effective > 0:
        scs_neurons = nf.create_input_neurons(num_scs_effective, scs_freq, 0, first_spike=scs_start)
        scs_pulse_times = nf.create_spike_recorder_input_neurons(scs_neurons)

    # Create a population of supraspinal neurons and record their spikes
    if num_supraspinal > 0: 
        supraspinal_neurons = nf.create_input_neurons(num_supraspinal, rate_supraspinal, noise=1, first_spike=0)
        supraspinal_spike_times = nf.create_spike_recorder_input_neurons(supraspinal_neurons)

    # Create a population of MNs and record their spikes
    mn_L = mn_avg_diameter + np.random.randn(num_mn)*0.1*mn_avg_diameter
    mns = [cll.MotoneuronNoDendrites("WT", drug=mn_drug, L=mn_L[imn]) for imn in range(num_mn)]
    mn_spike_times = nf.create_spike_recorder_mns(mns)
    
    # Connect a population of supraspinal fibers to MNs
    W_supraspinal = 0
    if num_supraspinal > 0:
        W_supraspinal = np.random.gamma(shape, scale=synaptic_weight_supra/shape, size=[num_supraspinal,num_mn])
        syn_supraspinal, nc_supraspinal = nf.create_exponential_synapses(supraspinal_neurons, mns, W_supraspinal, tau, inhibitory=supra_inhibit)

    # Connect a population of scs pulses to MNs
    W_scs = 0
    if num_scs_effective>0:
        W_scs = np.random.gamma(shape, scale=synaptic_weight_scs/shape, size=[num_scs_effective, num_mn])
        delay = np.random.lognormal(-0.47, 0.37, size=[num_scs_effective, num_mn]) # from Greiner 2021
        syn_scs, nc_scs = nf.create_exponential_synapses(scs_neurons, mns, W_scs, tau, delay)

    # Create stop command for supraspinal population
    stop_supra = h.NetStim()
    stop_supra.noise = 0 
    stop_supra.number = 1
    stop_supra.start = scs_start
   
    stop_supra_connections = [] 
    for i in range(num_supraspinal): 
        nc = h.NetCon(stop_supra, supraspinal_neurons[i])
        nc.weight[0] = -1
        stop_supra_connections.append(nc)

    # Create re-start command for supraspinal population
    re_start_supra = h.NetStim()
    re_start_supra.noise = 0 
    re_start_supra.number = 1
    re_start_supra.start = supraspinal_restart
   
    re_start_supra_connections = [] 
    for i in range(num_supraspinal): 
        nc = h.NetCon(re_start_supra, supraspinal_neurons[i])
        nc.weight[0] = 1
        re_start_supra_connections.append(nc)

    re_start_supra_times = h.Vector()
    re_start_supra_detector = h.NetCon(re_start_supra, None)
    re_start_supra_detector.record(re_start_supra_times)

    # Record motoneuron membranes
    membrane_potentials=[]
    for i in range(num_mn):
        membrane_potentials.append(h.Vector().record(mns[i].soma(0.5)._ref_v))
    simulation_time_vector = h.Vector().record(h._ref_t)

    # Run simulation
    print(f"Running simulation for {scs_amp} SCS amplitude, {scs_freq} SCS frequency, {num_supraspinal} supraspinal inputs") 
    h.finitialize()
    h.tstop = simulation_duration
    h.run()

    supraspinal_spike_times = [np.array(supraspinal_spike_times[i])  if len(supraspinal_spike_times[i]) > 0 else [] for i in range(num_supraspinal)]
    scs_pulse_times = [np.array(scs_pulse_times[i])  if len(scs_pulse_times[i]) > 0 else [] for i in range(num_scs_effective)]
    mn_spike_times = [np.array(mn_spike_times[i])  if len(mn_spike_times[i]) > 0 else [] for i in range(num_mn)]
    membrane_potentials = [np.array(membrane_potentials[i])  if len(membrane_potentials[i]) > 0 else [] for i in range(num_mn)]
    simulation_time_vector = np.array(simulation_time_vector)

    # Estimate EMG signal 
    emg_signal = nf.estimate_emg_signal(mn_spike_times, simulation_duration=simulation_duration)
    
    # Save simulation data
    if save_data_folder != '': 
        ensure_dir(save_data_folder)
        data_filename = f"mnNum_{num_mn}_supraspinalNum_{num_supraspinal}_supraspinalFR_{rate_supraspinal}_supraInhibit_{supra_inhibit}_SCSFreq_{scs_freq}_SCSTotal_{num_scs_total}_SCSAmp_{scs_amp}_SynW_{synaptic_weight_scs}.pickle"

        data={}
        data["mn_spikes"] = mn_spike_times
        data["supraspinal_spike_times"] = supraspinal_spike_times
        data["scs_pulse_times"] = scs_pulse_times
        data["scs_frequency"] = scs_freq
        data["scs_amp"] = scs_amp
        data["num_scs_total"] = num_scs_total
        data["W_scs"] = W_scs
        data["supraspinal_rate"] = rate_supraspinal
        data["num_supraspinal"] = num_supraspinal
        data["simulation_duration"] = simulation_duration
        data["num_mn"] = num_mn
        data["synaptic_weight_scs"] = synaptic_weight_scs
        data["synaptic_weight_supra"] = synaptic_weight_supra
        data["W_supraspinal"] = W_supraspinal
        data["mn_L"] = mn_L
        data["simulation_time_vector"] = simulation_time_vector
        data["membrane_potentials"] = membrane_potentials
        data["emg_signal"] = emg_signal

        f=open(save_data_folder+data_filename,"wb")
        pickle.dump(data,f)
        f.close()

    if plot_sim: 
        plt.subplots(3, 1, sharex='col')
        plt.suptitle(f'Motoneuron Membrane Potential:\n {scs_amp} SCS Amp, {scs_freq} SCS Freq,\n {num_supraspinal} Supraspinal Inputs', fontsize=8)

        # Plot motoneuron membrane potential
        ax = plt.subplot(3, 1, 1)
        pt.plot_time_series_data(ax, simulation_time_vector, membrane_potentials[0], ylabel='Motoneuron membrane\n potential (mV)')
        ax.set_ylim(-74, -60)

        # Plot supraspinal firing
        ax = plt.subplot(3, 1, 2)
        pt.plot_raster_plot(ax, supraspinal_spike_times, ylabel='Supraspinal\n fibers')
        ax.set_ylim([0, 300])

        # Plot SCS pulses
        ax = plt.subplot(3, 1, 3)
        pt.plot_raster_plot(ax, scs_pulse_times, ylabel='Ia-afferent\n input')
        ax.set_ylim([0, 60])
        ax.set_xlabel('Time (ms)', fontsize=8)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__': 
    scs_amp = 0.5
    scs_freq = 40
    perc_supra_intact = 0.2

    record_mn_membrane_potential_supraspinal_scs_simulation(scs_amp, 
                                                            scs_freq, 
                                                            perc_supra_intact, 
                                                            plot_sim=True)