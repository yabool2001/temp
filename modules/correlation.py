import csv
import numpy as np
from modules import plot
from numpy.typing import NDArray
from pathlib import Path

filename_results_csv = "correlation/correlation_results.csv"
base_path = Path ( filename_results_csv )

def correlation_v4 ( scenario ) :

    corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    
    corr_real = np.real ( corr_bpsk )
    corr_imag = np.imag ( corr_bpsk )
    corr_abs = np.abs ( corr_bpsk )
    
    # Znajdź wszystkie lokalne maksima (peaks) dla real, imag, abs
    def find_peaks(signal):
        # Prosta metoda znajdowania lokalnych maksimów: gdzie druga różnica jest ujemna
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        peaks = np.where(diff2 < 0)[0] + 1  # +1 bo diff przesuwa indeksy
        return peaks
    
    peaks_real = find_peaks(corr_real)
    peaks_imag = find_peaks(corr_imag)
    peaks_abs = find_peaks(corr_abs)
    
    name = f"{scenario[ 'desc' ]} | {scenario[ 'name' ]} {scenario[ 'mode' ]}"
    print ( f"{name}: peaks_real={peaks_real}, peaks_imag={peaks_imag}, peaks_abs={peaks_abs}" )
    plot.complex_waveform_v0_1_6 ( corr_bpsk , f"{name}" , False , peaks_abs )
    plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"{name}" , False , peaks_abs )
    plot.real_waveform_v0_1_6 ( corr_bpsk.real , f"{name}" , False , peaks_real )
    plot.real_waveform_v0_1_6 ( scenario[ "sample" ].real , f"{name}" , False , peaks_real )
    plot.real_waveform_v0_1_6 ( corr_bpsk.imag , f"{name}" , False , peaks_imag )
    plot.real_waveform_v0_1_6 ( scenario[ "sample" ].imag , f"{name}" , False , peaks_imag )


def correlation_v3 ( scenario ) :

    corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    
    corr_real = np.real ( corr_bpsk )
    corr_imag = np.imag ( corr_bpsk )
    corr_abs = np.abs ( corr_bpsk )
    
    peak_real_val = np.max ( np.abs ( corr_real ) )
    peak_imag_val = np.max ( np.abs ( corr_imag ) )
    peak_abs_val = np.max ( corr_abs )

    peak_real_idx = int ( np.argmax ( np.abs ( corr_real ) ) )
    peak_imag_idx = int ( np.argmax ( np.abs ( corr_imag ) ) )
    peak_abs_idx = int ( np.argmax ( corr_abs ) )
    
    print ( f"{scenario[ 'desc' ]}: {peak_real_idx=} {peak_real_val=}, {peak_imag_idx=} {peak_imag_val=}" )

    plot.complex_waveform_v0_1_6 ( corr_bpsk , f"{scenario[ 'desc' ]}" , False , np.array ( [ peak_abs_idx ] ) )
    plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"{scenario[ 'desc' ]}" , False , np.array ( [ peak_abs_idx ] ) )
    plot.real_waveform_v0_1_6 ( corr_bpsk.real , f"{scenario[ 'desc' ]}" , False , np.array ( [ peak_real_idx ] ) )
    plot.real_waveform_v0_1_6 ( scenario[ "sample" ].real , f"{scenario[ 'desc' ]}" , False , np.array ( [ peak_real_idx ] ) )
    plot.real_waveform_v0_1_6 ( corr_bpsk.imag , f"{scenario[ 'desc' ]}" , False , np.array ( [ peak_imag_idx ] ) )
    plot.real_waveform_v0_1_6 ( scenario[ "sample" ].imag , f"{scenario[ 'desc' ]}" , False , np.array ( [ peak_imag_idx ] ) )
        
    filename = base_path.parent / f"{scenario['desc']}_{base_path.name}"
    with open ( filename , 'w' , newline='' ) as csvfile :
        fieldnames = [ 'peak_abs_idx' , 'peak_abs_val' , 'peak_real_idx' , 'peak_real_val' , 'peak_imag_idx' , 'peak_imag_val' ]
        writer = csv.DictWriter ( csvfile , fieldnames = fieldnames )
        writer.writeheader ()
        writer.writerow (
            { 'peak_abs_idx': peak_abs_idx,
              'peak_abs_val': peak_abs_val,
              'peak_real_idx': peak_real_idx,
              'peak_real_val': peak_real_val,
              'peak_imag_idx': peak_imag_idx,
              'peak_imag_val': peak_imag_val }
        )

def correlation_v2 ( scenario , plt = True ) :

    corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    corr_real = np.real ( corr_bpsk )
    corr_imag = np.imag ( corr_bpsk )
    peak_real_val = np.max ( np.abs ( corr_real ) )
    peak_imag_val = np.max ( np.abs ( corr_imag ) )
    #if peak_real > peak_imag:
    peak_real_idx = int ( np.argmax ( np.abs ( corr_real ) ) )
    peak_imag_idx = int ( np.argmax ( np.abs ( corr_imag ) ) )
    name = f"{scenario[ 'desc' ]} | {scenario[ 'name' ]} {scenario[ 'mode' ]}"
    print ( f"{name}: {peak_real_idx=} {peak_real_val=}, {peak_imag_idx=} {peak_imag_val=}" )
    if plt :
        plot.complex_waveform_v0_1_6 ( corr_bpsk , f"{name}" , False , np.array ( [ peak_real_idx , peak_imag_idx ] ) )
        plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"{name}" , False , np.array ( [ peak_real_idx , peak_imag_idx ] ) )
        plot.real_waveform_v0_1_6 ( corr_bpsk.real , f"{name}" , False , np.array ( [ peak_real_idx ] ) )
        plot.real_waveform_v0_1_6 ( scenario[ "sample" ].real , f"{name}" , False , np.array ( [ peak_real_idx ] ) )
        plot.real_waveform_v0_1_6 ( corr_bpsk.imag , f"{name}" , False , np.array ( [ peak_imag_idx ] ) )
        plot.real_waveform_v0_1_6 ( scenario[ "sample" ].imag , f"{name}" , False , np.array ( [ peak_imag_idx ] ) )

    filename = base_path.parent / f"{scenario['name']}_{base_path.name}"
    with open ( filename , 'w' , newline='' ) as csvfile :
        fieldnames = [ 'peak_real_idx' , 'peak_real_val' , 'peak_imag_idx' , 'peak_imag_val' ]
        writer = csv.DictWriter ( csvfile , fieldnames = fieldnames )
        writer.writeheader ()
        writer.writerow ( { 'peak_real_idx': peak_real_idx, 'peak_real_val': peak_real_val, 'peak_imag_idx': peak_imag_idx, 'peak_imag_val': peak_imag_val } )

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