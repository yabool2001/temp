import csv
import numpy as np
import time as t
from modules import modulation , plot
from numpy.typing import NDArray
from pathlib import Path
from scipy.signal import find_peaks

filename_results_csv = "correlation/correlation_results.csv"
base_path = Path ( filename_results_csv )

plt = True
wrt = True


def correlation_v8 ( scenario ) :

    peaks = np.array ( [] )
    sync = False
    t0 = t.perf_counter_ns ()

    max_amplitude = np.max ( np.abs ( scenario [ 'sample' ] ) )

    corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    
    corr_real = np.abs ( np.real ( corr_bpsk ) )
    corr_imag = np.abs ( np.imag ( corr_bpsk ) )
    corr_abs = np.abs ( corr_bpsk )

    max_peak_real_val = np.max ( corr_real )
    max_peak_imag_val = np.max ( corr_imag )
    max_peak_abs_val = np.max ( corr_abs )

    corr_2_amp = np.max ( [ max_peak_real_val , max_peak_imag_val , max_peak_abs_val ] ) / max_amplitude

    if corr_2_amp > 12 :
        sync = True
        # Znajdź peaks powyżej threshold i z prominence dla real, imag, abs
        #peaks_real, _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS , prominence = 0.5 )
        peaks_real , _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
        peaks_imag , _ = find_peaks ( corr_imag , height = max_peak_imag_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
        peaks_abs , _ = find_peaks ( corr_abs , height = max_peak_abs_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
        peaks = np.unique ( np.concatenate ( ( peaks_real , peaks_imag , peaks_abs ) ) )
    else :
        print ( f"Nie ma korelacji! {corr_2_amp=}" )

    t1 = t.perf_counter_ns ()
    print ( f"Correlation scenario_old2_nc: {scenario['desc']} took {(t1 - t0)/1e3:.1f} µs" )
    
    if plt and sync :
        plot.real_waveform_v0_1_6 ( corr_abs , f"V7 corr abs {scenario[ 'desc' ]}" , False , peaks_abs )
        plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"V7 samples abs {scenario[ 'desc' ]}" , False , peaks_abs )
        plot.real_waveform_v0_1_6 ( corr_real , f"V7 corr real {scenario[ 'desc' ]}" , False , peaks_real )
        plot.real_waveform_v0_1_6 ( scenario[ "sample" ].real , f"V7 samples real {scenario[ 'desc' ]}" , False , peaks_real )
        plot.real_waveform_v0_1_6 ( corr_imag , f"V7 corr imag {scenario[ 'desc' ]}" , False , peaks_imag )
        plot.real_waveform_v0_1_6 ( scenario[ "sample" ].imag , f"V7 samples imag {scenario[ 'desc' ]}" , False , peaks_imag )
        plot.complex_waveform_v0_1_6 ( corr_bpsk , f"V7 corr all {scenario[ 'desc' ]}" , False , peaks )
        plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"V7 samples all {scenario[ 'desc' ]}" , False , peaks )

    if wrt and sync:
        filename = base_path.parent / f"V7_{scenario['desc']}_{base_path.name}"
        with open ( filename , 'w' , newline='' ) as csvfile :
            fieldnames = ['corr', 'peak_idx', 'peak_val']
            writer = csv.DictWriter ( csvfile , fieldnames = fieldnames )
            writer.writeheader ()
            for idx in peaks_abs :
                writer.writerow ( { 'corr': 'abs' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_abs[ idx ] ) } )
            for idx in peaks_real :
                writer.writerow ( { 'corr' : 'real' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_real[ idx ] ) } )
            for idx in peaks_imag :
                writer.writerow ( { 'corr' : 'imag' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_imag[ idx ] ) } )
            for idx in peaks :
                writer.writerow ( { 'corr' : 'all' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_abs[ idx ] ) } )

        return peaks

def correlation_v7 ( scenario ) :

    t0 = t.perf_counter_ns ()

    corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    
    corr_real = np.abs ( np.real ( corr_bpsk ) )
    corr_imag = np.abs ( np.imag ( corr_bpsk ) )
    corr_abs = np.abs ( corr_bpsk )

    max_peak_real_val = np.max ( corr_real )
    max_peak_imag_val = np.max ( corr_imag )
    max_peak_abs_val = np.max ( corr_abs )

    # Znajdź peaks powyżej threshold i z prominence dla real, imag, abs
    #peaks_real, _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS , prominence = 0.5 )
    peaks_real , _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
    peaks_imag , _ = find_peaks ( corr_imag , height = max_peak_imag_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
    peaks_abs , _ = find_peaks ( corr_abs , height = max_peak_abs_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )

    peaks = np.unique ( np.concatenate ( ( peaks_real , peaks_imag , peaks_abs ) ) )
    t1 = t.perf_counter_ns ()
    print ( f"Correlation scenario_old2_nc: {scenario['desc']} took {(t1 - t0)/1e3:.1f} µs" )
    
    if plt :
        plot.real_waveform_v0_1_6 ( corr_abs , f"V7 corr abs {scenario[ 'desc' ]}" , False , peaks_abs )
        plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"V7 samples abs {scenario[ 'desc' ]}" , False , peaks_abs )
        plot.real_waveform_v0_1_6 ( corr_real , f"V7 corr real {scenario[ 'desc' ]}" , False , peaks_real )
        plot.real_waveform_v0_1_6 ( scenario[ "sample" ].real , f"V7 samples real {scenario[ 'desc' ]}" , False , peaks_real )
        plot.real_waveform_v0_1_6 ( corr_imag , f"V7 corr imag {scenario[ 'desc' ]}" , False , peaks_imag )
        plot.real_waveform_v0_1_6 ( scenario[ "sample" ].imag , f"V7 samples imag {scenario[ 'desc' ]}" , False , peaks_imag )
        plot.complex_waveform_v0_1_6 ( corr_bpsk , f"V7 corr all {scenario[ 'desc' ]}" , False , peaks )
        plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"V7 samples all {scenario[ 'desc' ]}" , False , peaks )

    if wrt :
        filename = base_path.parent / f"V7_{scenario['desc']}_{base_path.name}"
        with open ( filename , 'w' , newline='' ) as csvfile :
            fieldnames = ['corr', 'peak_idx', 'peak_val']
            writer = csv.DictWriter ( csvfile , fieldnames = fieldnames )
            writer.writeheader ()
            for idx in peaks_abs :
                writer.writerow ( { 'corr': 'abs' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_abs[ idx ] ) } )
            for idx in peaks_real :
                writer.writerow ( { 'corr' : 'real' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_real[ idx ] ) } )
            for idx in peaks_imag :
                writer.writerow ( { 'corr' : 'imag' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_imag[ idx ] ) } )
            for idx in peaks :
                writer.writerow ( { 'corr' : 'all' , 'peak_idx' : int ( idx ) , 'peak_val' : float ( corr_abs[ idx ] ) } )


def correlation_v6 ( scenario ) :

    corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    
    corr_real = np.real ( corr_bpsk )
    corr_imag = np.imag ( corr_bpsk )
    corr_abs = np.abs ( corr_bpsk )
    
    # Oblicz threshold na podstawie sync_sequence
    L = len(scenario["sync_sequence"])
    ideal_peak = L * np.sqrt(np.mean(np.abs(scenario["sync_sequence"])**2))
    threshold = 0.9 * ideal_peak
    prominence = 0.5 * ideal_peak
    
    # Znajdź peaks powyżej threshold i z prominence dla real, imag, abs
    peaks_real, _ = find_peaks(corr_real, height=threshold, prominence=prominence)
    peaks_imag, _ = find_peaks(corr_imag, height=threshold, prominence=prominence)
    peaks_abs, _ = find_peaks(corr_abs, height=threshold, prominence=prominence)
    
    # Zapisz wszystkie peaks do CSV
    filename = base_path.parent / f"V6_{scenario['desc']}_{base_path.name}"
    with open ( filename , 'w' , newline='' ) as csvfile :
        fieldnames = ['corr', 'peak_idx', 'peak_val']
        writer = csv.DictWriter ( csvfile , fieldnames = fieldnames )
        writer.writeheader ()
        
        # Dodaj wiersze dla peaks_abs
        for idx in peaks_abs:
            writer.writerow ({
                'corr': 'abs',
                'peak_idx': int(idx),
                'peak_val': float(corr_abs[idx])
            })
        
        # Dodaj wiersze dla peaks_real
        for idx in peaks_real:
            writer.writerow ({
                'corr': 'real',
                'peak_idx': int(idx),
                'peak_val': float(corr_real[idx])
            })
        
        # Dodaj wiersze dla peaks_imag
        for idx in peaks_imag:
            writer.writerow ({
                'corr': 'imag',
                'peak_idx': int(idx),
                'peak_val': float(corr_imag[idx])
            })
    
def correlation_v5 ( scenario ) :

    corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    
    corr_real = np.abs ( np.real ( corr_bpsk ) )
    corr_imag = np.abs ( np.imag ( corr_bpsk ) )
    corr_abs = np.abs ( corr_bpsk )

    max_peak_real_val = np.max ( corr_real )
    max_peak_imag_val = np.max ( corr_imag )
    max_peak_abs_val = np.max ( corr_abs )

    # Znajdź peaks powyżej threshold i z prominence dla real, imag, abs
    #peaks_real, _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS , prominence = 0.5 )
    peaks_real , _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
    peaks_imag , _ = find_peaks ( corr_imag , height = max_peak_imag_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
    peaks_abs , _ = find_peaks ( corr_abs , height = max_peak_abs_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
    
    name = f"{scenario[ 'desc' ]} | {scenario[ 'name' ]} {scenario[ 'mode' ]}"
    print ( f"{name}: peaks_real={peaks_real}, peaks_imag={peaks_imag}, peaks_abs={peaks_abs}" )
    plot.real_waveform_v0_1_6 ( corr_abs , f"{name}" , False , peaks_abs )
    plot.complex_waveform_v0_1_6 ( scenario[ "sample" ] , f"{name}" , False , peaks_abs )
    plot.real_waveform_v0_1_6 ( corr_real , f"{name}" , False , peaks_real )
    plot.real_waveform_v0_1_6 ( scenario[ "sample" ].real , f"{name}" , False , peaks_real )
    plot.real_waveform_v0_1_6 ( corr_imag , f"{name}" , False , peaks_imag )
    plot.real_waveform_v0_1_6 ( scenario[ "sample" ].imag , f"{name}" , False , peaks_imag )

    filename = base_path.parent / f"V5_{scenario['desc']}_{base_path.name}"
    with open ( filename , 'w' , newline='' ) as csvfile :
        fieldnames = [ 'peaks_abs_idx' , 'peaks_real_idx' , 'peaks_imag_idx' ,]
        writer = csv.DictWriter ( csvfile , fieldnames = fieldnames )
        writer.writeheader ()
        writer.writerow ( {
            'peaks_abs_idx': peaks_abs ,
            'peaks_real_idx': peaks_real ,
            'peaks_imag_idx': peaks_imag
        } )


def correlation_v4 ( scenario ) :

    corr_bpsk = np.correlate ( scenario[ "sample" ] , scenario[ "sync_sequence" ] , mode = scenario[ "mode" ] )
    
    corr_real = np.real ( corr_bpsk )
    corr_imag = np.imag ( corr_bpsk )
    corr_abs = np.abs ( corr_bpsk )

    max_peak_real_val = np.max ( np.abs ( corr_real ) )
    max_peak_imag_val = np.max ( np.abs ( corr_imag ) )
    max_peak_abs_val = np.max ( corr_abs )

    # Znajdź peaks powyżej threshold i z prominence dla real, imag, abs
    #peaks_real, _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS , prominence = 0.5 )
    peaks_real, _ = find_peaks ( corr_real , height = max_peak_real_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
    peaks_imag, _ = find_peaks ( corr_imag , height = max_peak_imag_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
    peaks_abs, _ = find_peaks ( corr_abs , height = max_peak_abs_val - max_peak_real_val * 0.1 , distance = 13 * modulation.SPS )
    
    # Sprawdź wiarygodność synchronizacji na podstawie zakresu peaks
    if len(peaks_abs) > 0 and np.ptp(peaks_abs) <= 4:
        print("Synchronizacja potwierdzona!")
        # Tutaj można dodać logikę wycinania danych od peak_idx, np. od peaks_abs[0]
    else:
        print("Brak wiarygodnej synchronizacji")
    
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