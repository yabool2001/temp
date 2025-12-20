import csv
import numpy as np
from modules import plot
from numpy.typing import NDArray
from pathlib import Path

filename_results_csv = "correlation/correlation_results.csv"

def correlation_v2 ( scenarios , plt = True ) :

    results = []
    for scenario in scenarios:
        corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
        corr_real = np.real ( corr_bpsk )
        corr_imag = np.imag ( corr_bpsk )
        peak_real = np.max ( np.abs ( corr_real ) )
        peak_imag = np.max ( np.abs ( corr_imag ) )
        peak_val = max ( peak_real , peak_imag )
        if peak_real > peak_imag:
            peak_idx = int ( np.argmax ( np.abs ( corr_real ) ) )
        else:
            peak_idx = int ( np.argmax ( np.abs ( corr_imag ) ) )
        peak_phase = np.angle ( corr_bpsk[ peak_idx ] )
        name = f"{scenario[ 'desc' ]} | {scenario[ 'name' ]} {scenario[ 'mode' ]}"
        print ( f"{name}: {peak_idx=}, {peak_val=}, {peak_phase=}" )
        results.append ( {
            'name' : scenario['name'],
            'description' : scenario['desc'],
            'correlation_mode' : scenario['mode'],
            'peak_idx' : peak_idx,
            'peak_val' : peak_val
        } )
        if plt :
            plot.complex_waveform_v0_1_6 ( corr_bpsk , f"{name}" , False , np.array ( [ peak_idx ] ) )
            plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"{name}" , False , np.array ( [ peak_idx ] ) )
        

    with open ( filename_results_csv , 'w' , newline='' ) as csvfile :
        fieldnames = ['name', 'description', 'correlation_mode', 'conjugate', 'flip', 'magnitude', 'peak_idx', 'peak_val']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def correlation_v1 ( scenarios , plt = True ) :

    results = []
    for scenario in scenarios:
        corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
        corr_real = np.real ( corr_bpsk )
        corr_imag = np.imag ( corr_bpsk )
        peak_real = np.max ( np.abs ( corr_real ) )
        peak_imag = np.max ( np.abs ( corr_imag ) )
        peak_val = max ( peak_real , peak_imag )
        if peak_real > peak_imag:
            peak_idx = int ( np.argmax ( np.abs ( corr_real ) ) )
        else:
            peak_idx = int ( np.argmax ( np.abs ( corr_imag ) ) )
        peak_phase = np.angle ( corr_bpsk[ peak_idx ] )
        name = f"{scenario[ 'desc' ]} | {scenario[ 'name' ]} {scenario[ 'mode' ]}"
        print ( f"{name}: {peak_idx=}, {peak_val=}, {peak_phase=}" )
        results.append ( {
            'name' : scenario['name'],
            'description' : scenario['desc'],
            'correlation_mode' : scenario['mode'],
            'peak_idx' : peak_idx,
            'peak_val' : peak_val
        } )
        if plt :
            plot.complex_waveform_v0_1_6 ( corr_bpsk , f"{name}" , True , np.array ( [ peak_idx ] ) )
            plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"{name}" , True , np.array ( [ peak_idx ] ) )
        

    with open ( filename_results_csv , 'w' , newline='' ) as csvfile :
        fieldnames = ['name', 'description', 'correlation_mode', 'conjugate', 'flip', 'magnitude', 'peak_idx', 'peak_val']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)