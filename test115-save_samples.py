from modules import corrections , filters , modulation , ops_file , ops_packet , plot , sdr
from pathlib import Path
import numpy as np
import os
import tomllib
from numpy.typing import NDArray

Path ( "np.samples" ).mkdir ( parents = True , exist_ok = True )

plt = True
wrt = True

filename = "np.samples/rx_samples_1.npy"

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

pluto_rx = sdr.init_pluto_v3 ( settings["ADALM-Pluto"]["URI"]["SN_RX"] )
for i in range ( 0 , 1 ) : # Clear buffer just to be safe
    raw_data = sdr.rx_samples ( pluto_rx )
print ( "Start Pluto Rx!" )
work = True
while work :
    rx_samples = sdr.rx_samples ( pluto_rx )
    rx_samples_filtered = filters.apply_rrc_rx_filter_v0_1_6 ( rx_samples )
    rx_samples_corrected = corrections.full_compensation_v0_1_5 ( rx_samples_filtered , modulation.get_barker13_bpsk_samples_v0_1_3 ( True ) )
    rx_samples_corrected = modulation.zero_quadrature ( rx_samples_corrected )
    corr = modulation.fast_normalized_cross_correlation ( rx_samples_corrected , modulation.get_barker13_bpsk_samples_v0_1_3 ( True ) )
    threshold = np.mean ( corr ) + 3 * np.std ( corr )
    detected_peaks = np.where(corr >= threshold)[0]
    peaks = modulation.group_peaks_by_distance ( detected_peaks , corr , min_distance = 2 )
    for peak in peaks :
        if ops_packet.is_preamble ( rx_samples_corrected[ peak: ] , filters.SPAN , modulation.SPS ) :
            print ( "Preambuła znaleziona!" )
            #if plt :
            #    plot.complex_waveform_v0_1_6 ( rx_samples , f"{rx_samples.size=}" , False )
            #if wrt :
            #    ops_file.save_complex_samples_2_npf ( filename , rx_samples )
            work = False
if plt :
    plot.complex_waveform_v0_1_6 ( rx_samples , f"{rx_samples.size=}" , False )
if wrt :
    ops_file.save_complex_samples_2_npf ( filename , rx_samples )
        