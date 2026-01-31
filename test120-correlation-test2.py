'''
Docstring for test120-correlation-test2.py

To jest rozszerzenie skryptu test120-correlation-test.py w którym testowano metodę znormalizowanej korelacji wzajemnej (Normalized Cross-Correlation).
Pozwala ona na wykrywanie wzorca (sekwencji synchronizacyjnej) nawet w fragmentach sygnału o bardzo niskiej amplitudzie,
ponieważ każda "karetka" korelacji jest dzielona przez energię sygnału znajdującego się aktualnie w oknie.

To jest rozszerzenie skryptu test120-correlation-test.py o testy na sygnale real i imag z rzeczywistego przebiegu IQ z ADALM-Pluto z 20 ramkami z wzorcem sync_sequence
ale o coraz mniejszej amplitudzie. Jednocześenie usunąłem proste sygnały z wcześniejszgo przykładu.
'''

from modules import modulation, ops_file , plot
import numpy as np
from numpy.typing import NDArray
import os
from scipy.signal import find_peaks
import time as t
import csv

script_filename = os.path.basename ( __file__ )

plt = False

filename_sync_sequence_1 = "correlation/test120_sync_sequence.npy"
filename_samples_2_1_3_4_npy = "correlation/test120_samples_2_1_3_4.npy"
filename_samples_128Bx20_npy = "np.samples/rx_samples_0.1.14_128Bx20_missed_last_frames.npy"

sync_sequence_1 = ops_file.open_real_float64_samples_from_npf ( filename_sync_sequence_1 )
samples_2_1_3_4  = ops_file.open_real_float64_samples_from_npf ( filename_samples_2_1_3_4_npy )
sync_sequence_2 = modulation.generate_barker13_bpsk_samples_v0_1_7 ( True )
samples_128Bx20  = ops_file.open_samples_from_npf ( filename_samples_128Bx20_npy )

if plt :
    plot.real_waveform_v0_1_6 ( sync_sequence_1 , f"{sync_sequence_1.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_2_1_3_4 , f"{samples_2_1_3_4.size=}" , False )
    plot.real_waveform_v0_1_6 ( sync_sequence_2.real , f"{sync_sequence_2.size=}" , False )
    plot.complex_waveform_v0_1_6 ( samples_128Bx20 , f"{samples_128Bx20.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_128Bx20.real , f"{samples_128Bx20.real.size=}" , False )
    plot.real_waveform_v0_1_6 ( samples_128Bx20.imag , f"{samples_128Bx20.imag.size=}" , False )

scenarios = [
    { "name" : "samples_2_1_3_4" , "desc" : "combined samples 2,1,3,4" , "samples" : samples_2_1_3_4 , "sync_sequence" : sync_sequence_1 } ,
    { "name" : "128Bx20 real" , "desc" : "real part of 128 bytes x 20 times" , "samples" : samples_128Bx20.real , "sync_sequence" : sync_sequence_2.real } ,
    { "name" : "128Bx20 imag" , "desc" : "imag part of 128 bytes x 20 times" , "samples" : samples_128Bx20.imag , "sync_sequence" : sync_sequence_2.real }
]

def my_correlation ( scenario : dict ) -> None :

    min_peak_height_ratio = 0.8

    sync_sequence : NDArray[ np.float64 ] = scenario["sync_sequence"]
    samples : NDArray[ np.float64 ] = scenario["samples"]
    samples_normalized : NDArray[ np.float64 ] = ( samples / np.max ( np.abs ( samples ) ) * 2 )
    plot.real_waveform_v0_1_6 ( samples_normalized , f"samples normalized {scenario['name']} {samples_normalized.size=}" , False )

    peaks_normalized = np.array ( [] ).astype ( np.uint32 )

    corr = np.correlate ( samples , sync_sequence , mode = "valid" )
    corr = np.abs ( corr ) # niezbędne

    #corr_normalized = np.correlate ( samples_normalized , sync_sequence , mode = "valid" )

    '''Znormalizowaną Korelacją Wzajemną (Normalized Cross-Correlation).
    Pozwala ono na wykrywanie wzorca (sekwencji synchronizacyjnej) nawet w fragmentach sygnału o bardzo niskiej amplitudzie,
    ponieważ każda "karetka" korelacji jest dzielona przez energię sygnału znajdującego się aktualnie w oknie.

    Zalety tego rozwiązania:

    Odporność na zmiany amplitudy: Wykryjesz sekwencję tak samo dobrze, gdy sygnał jest bardzo cichy,
    jak i gdy jest bardzo głośny (w ramach tego samego pliku samples).
    Stały próg detekcji: Możesz ustawić próg height w find_peaks na stałą wartość, np. 0.8 (oznaczającą 80% podobieństwa kształtu),
    bez zgadywania amplitudy.

    Funkcja np.correlate nie ma wbudowanej opcji "lokalnej normalizacji", ale można ją bardzo szybko obliczyć "na boku",
    wykorzystując trick z drugą korelacją (filtrem średniej ruchomej) na kwadratach próbek. Poniżej znajduje się implementacja tego podejścia.
    '''
    # Obliczamy lokalną energię sygnału (sliding sum of squares)
    # Splot kwadratów sygnału z oknem z samych jedynek daje sumę energii w oknie
    ones = np.ones ( len ( sync_sequence ) )
    local_energy = np.correlate ( samples**2 , ones , mode = "valid" )
    # Obliczamy normy do mianownika
    # Norma sekwencji synchronizacyjnej (stała skalarna)
    sync_seq_norm = np.linalg.norm ( sync_sequence )
    # Lokalna norma sygnału (wektor o długości wyniku korelacji)
    # Dodajemy epsilon (np. 1e-10) lub maximum, aby uniknąć dzielenia przez zero w ciszy
    local_signal_norm = np.sqrt ( np.maximum ( local_energy , 1e-10 ) )
    # Wynik znormalizowany (wartości teoretycznie od -1.0 do 1.0)
    corr_normalized = corr / ( local_signal_norm * sync_seq_norm )

    max_peak_val_normalized = np.max ( corr_normalized )

    peaks_normalized , _ = find_peaks ( corr_normalized , height = max_peak_val_normalized * min_peak_height_ratio )

    plot.real_waveform_v0_1_6 ( corr_normalized , f"corr normalized2 {scenario['name']} {corr_normalized.size=} {peaks_normalized.size=}" , False , peaks_normalized )
    plot.real_waveform_v0_1_6 ( samples , f"samples normalized2 {scenario['name']} {samples.size=} {peaks_normalized.size=}" , False , peaks_normalized )


#my_correlation ( scenarios[4] )
for scenario in scenarios :
    my_correlation ( scenario )