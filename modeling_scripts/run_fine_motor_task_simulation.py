from matplotlib import pyplot as plt
from neuron import h
import numpy as np
import pickle

import sys
sys.path.append('../SCSinSCIBiophysicalModel')
from tools.general_tools import ensure_dir
import cells as cll
import tools.neuron_functions as nf
import tools.analysis_tools as at
import tools.plotting_tools as pt

def run_fine_motor_task_simulation(scs_amp: float,
                                   scs_freq: int, 
                                   perc_supra_intact: float = 1, 
                                   supraspinal_inhomogenous_rate: int = 0.001,
                                   simulation_duration: int = 4000,
                                   save_data_folder: str = '', 
                                   plot_sim: bool = False):
    """ 
    Run simulation of a fine motor task, in which motoneuron receive 
    continuous SCS input but varying supraspinal inputs (inhomogenous Poisson). 

    Args: 
        - scs_amp: float, amplitude of stimulation (% of max # of SCS inputs)
        - scs_freq: int, frequency of stimulation (Hz)
        - perc_supra_intact: float, remaining supraspinal inputs (% of intact model = 300)
        - supraspinal_inhomogenous_rate: int, rate of sinusoidal wave for inhomogenous Poisson supraspinal input (Hz)
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
    num_mn = 100
    mn_drug = True
    mn_avg_diameter = 36
    
    # Set synaptic parameters
    synaptic_weight = 0.000148
    shape = 1.2
    tau = 2

    # Create lists to hold the neurons and spike recorders
    supraspinal_neurons = []
    scs_neurons = []
    supraspinal_spike_times = []
    scs_pulse_times = []

    # Create a population of SCS pulses and record their spikes
    if num_scs_effective > 0:
        scs_neurons = nf.create_input_neurons(num_scs_effective,scs_freq,0)
        scs_pulse_times = nf.create_spike_recorder_input_neurons(scs_neurons)

    # Create a population of inhomogeneous supraspinal pulses and record its spikes
    if num_supraspinal > 0: 
        supraspinal_neurons = nf.create_inhomogeneous_input_neurons(num_supraspinal, rate_supraspinal, simulation_duration, frequency=supraspinal_inhomogenous_rate)
        supraspinal_spike_times = nf.create_spike_recorder_input_neurons(supraspinal_neurons)

    # Create a population of MNs and record their spikes
    mn_L = mn_avg_diameter + np.random.randn(num_mn)*0.1*mn_avg_diameter
    mns = [cll.MotoneuronNoDendrites("WT", drug=mn_drug, L=mn_L[imn]) for imn in range(num_mn)]
    mn_spike_times = nf.create_spike_recorder_mns(mns)

    # Connect a population of supraspinal fibers to MNs
    W_supraspinal = 0
    if num_supraspinal > 0:
        W_supraspinal = np.random.gamma(shape, scale=synaptic_weight/shape, size=[num_supraspinal,num_mn])
        syn_supraspinal, nc_supraspinal = nf.create_exponential_synapses(supraspinal_neurons, mns, W_supraspinal, tau)

    # Connect a population of scs pulses to MNs
    W_scs = 0
    if num_scs_effective>0:
        W_scs = np.random.gamma(shape, scale=synaptic_weight/shape, size=[num_scs_effective, num_mn])
        delay = np.random.lognormal(-0.47, 0.37, size=[num_scs_effective, num_mn]) # from Greiner 2021
        syn_scs, nc_scs = nf.create_exponential_synapses(scs_neurons, mns, W_scs, tau, delay)

    # Run simulation
    print(f"Running simulation for {scs_amp} SCS amplitude, {scs_freq} SCS frequency, {num_supraspinal} supraspinal inputs") 
    h.finitialize()
    h.tstop = simulation_duration
    h.run()

    # Convert outputs to numpy
    supraspinal_spike_times = [np.array(supraspinal_spike_times[i])  if len(supraspinal_spike_times[i]) > 0 else [] for i in range(num_supraspinal)]
    scs_pulse_times = [np.array(scs_pulse_times[i])  if len(scs_pulse_times[i]) > 0 else [] for i in range(num_scs_effective)]
    mn_spike_times = [np.array(mn_spike_times[i])  if len(mn_spike_times[i]) > 0 else [] for i in range(num_mn)]
    
    # Estimate EMG signal 
    emg_signal = nf.estimate_emg_signal(mn_spike_times, simulation_duration=simulation_duration)
    
    # Save simulation data
    if save_data_folder != '': 
        ensure_dir(save_data_folder)
        data_filename = f"mnNum_{num_mn}_supraspinalNum_{num_supraspinal}_supraspinalFR_{rate_supraspinal}_SCSFreq_{scs_freq}_SCSTotal_{num_scs_total}_SCSAmp_{scs_amp}_SynW_{synaptic_weight}.pickle"

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
        data["synaptic_weight_scs"] = synaptic_weight
        data["synaptic_weight_supra"] = synaptic_weight
        data["W_supraspinal"] = W_supraspinal
        data["mn_L"] = mn_L
        data["emg_signal"] = emg_signal

        f=open(save_data_folder+data_filename,"wb")
        pickle.dump(data,f)
        f.close()

    # Visualize simulation data
    if plot_sim: 
        plt.subplots(3, 1, sharex='col')
        plt.suptitle(f'Fine Motor Task:\n {scs_amp} SCS Amp, {scs_freq} SCS Freq,\n {num_supraspinal} Supraspinal Inputs', fontsize=8)

        # Plot generated force
        ax = plt.subplot(3, 1, 1)
        force = at.firing_rate_to_force(mn_spike_times, simulation_duration)
        pt.plot_time_series_data(ax, [t for t in range(len(force))], force, 'Force (Nm)')

        bin_size = 100
        binned_time_vector = [x for x in range(int(bin_size/2), simulation_duration-int(bin_size/2), bin_size)]
        # Plot motoneuron binned firing rate (Hz)
        ax = plt.subplot(3, 1, 2)
        mn_fr = at.bin_fr_hz(mn_spike_times, simulation_duration, bin_size=bin_size)    
        pt.plot_time_series_data(ax, binned_time_vector, mn_fr, ylabel='Motoneuron firing\n rate (Hz)')

        # Plot supraspinal binned firing rate (Hz)
        ax = plt.subplot(3, 1, 3)
        supra_fr = at.bin_fr_hz(supraspinal_spike_times, simulation_duration, bin_size=bin_size)
        pt.plot_time_series_data(ax, binned_time_vector, supra_fr, ylabel='Supraspinal firing\n rate (Hz)')
        ax.set_xlabel('Time (ms)', fontsize=8)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    scs_amp = 0.5
    scs_freq = 40
    perc_supra_intact = 0.2

    run_fine_motor_task_simulation(scs_amp, 
                                    scs_freq, 
                                    perc_supra_intact, 
                                    plot_sim=True)