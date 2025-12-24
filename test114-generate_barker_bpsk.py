from modules import filters , ops_packet , ops_file , plot , modulation
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

Path ( "correlation" ).mkdir ( parents = True , exist_ok = True )

plt = True
wrt = False

def generate_barker13_bpsk_samples_v0_1_3 ( clipped = False ) -> NDArray[ np.complex128 ] :
    symbols = modulation.create_bpsk_symbols ( ops_packet.BARKER13_BITS )
    samples = filters.apply_tx_rrc_filter_v0_1_3 ( symbols , True )
    if clipped :
        tail_length = ( filters.SPAN - 1 ) // 2 * modulation.SPS  # Oblicz ogon filtra RRC
        samples = samples[ : -tail_length ]  # Przytnij ogon na końcu
    return samples

barker13 = generate_barker13_bpsk_samples_v0_1_3 ( clipped = False )
barker13_clipped = generate_barker13_bpsk_samples_v0_1_3 ( clipped = True )

if plt :
    plot.complex_waveform_v0_1_6 ( barker13 , f"{barker13.size=}" , True )
    plot.complex_waveform_v0_1_6 ( barker13_clipped , f"{barker13_clipped.size=}" , True )

if wrt :
    #ops_file.save_complex_samples_2_npf ( filename_barker13 , barker13 )
    #ops_file.save_complex_samples_2_npf ( filename_barker13_clipped , barker13_clipped )
    #ops_file.save_samples_2_npf ( filename_sync_sequence_1 , sync_sequence_1 )
    #ops_file.save_samples_2_npf ( filename_sync_sequence_2 , sync_sequence_2 )
    #ops_file.save_samples_2_npf ( filename_samples_1 , samples_1 )
    #ops_file.save_samples_2_npf ( filename_samples_2 , samples_2 )
    pass