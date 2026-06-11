from pathlib import Path
import tomllib

import numpy as np , os
import torch
from numpy.typing import NDArray

from modules import filters, modulation, packet , plot , ops_file

script_filename = os.path.basename ( __file__ )
# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )
    
dbg : bool = True
plt : bool = True
obj : bool = True # obj=object czy chcesz to robic za pomoca klas w modules/packet.py czy tylko funkcji w modules/ops_file.py

rx_samples_file_name = "np.test/1781186741813_rx_samples_1781186742276.npy"
rx_samples = packet.RxSamples ()
rx_samples.rx ( file_name = str ( rx_samples_file_name ) , concatenate = False )
rx_samples.detect_frames ( deep = False , samples_filtered = True , correct_samples = False )
idxs = rx_samples.aggregate_frame_and_packet_idxs ()

tx_samples_file_name = "np.test/1781186741813_tx_samples.npy"
tx_active_samples_file_name = "np.test/1781186741813_tx_active_samples.npy"
tx_symbols_file_name = "np.test/1781186741813_tx_symbols.npy"
tx_samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( tx_samples_file_name ) )
tx_active_samples : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( tx_active_samples_file_name ) )
tx_symbols : NDArray[ np.complex128 ] = ops_file.open_samples_from_npf ( str ( tx_symbols_file_name ) )

timestamp_group = rx_samples_file_name.split ( "_rx_samples" , 1 )[ 0 ]
radio_preamble_bytes = np.array ( toml_settings[ "RADIO_PREAMBLE_BYTES" ] , dtype = np.uint8 )
radio_preamble_bits : NDArray [ np.complex128 ] = packet.bytes2bits ( radio_preamble_bytes )
radio_preamble_bpsk_symbols_len : np.uint32 = radio_preamble_bits.size  * modulation.SPS

#rx_samples_first_symbol_abs_idx = idxs[0] - radio_preamble_bpsk_symbols_len - filters.FIRST_SYMBOL_OFFSET
rx_samples_first_symbol_abs_idx = idxs[0]
rx_samples_last_symbol_abs_idx = rx_samples_first_symbol_abs_idx + tx_symbols.size

plot.samples_and_tensor_1k ( X_train_samples = rx_samples.samples_raw[ rx_samples_first_symbol_abs_idx : rx_samples_last_symbol_abs_idx ] ,
							y_train_tensor = tx_samples[ filters.FIRST_SYMBOL_OFFSET : ] ,
							ai_samples = tx_active_samples ,
							ai_symbols = tx_symbols ,
							idxs = idxs - rx_samples_first_symbol_abs_idx ,
							my_title = f"{script_filename} {timestamp_group}" )