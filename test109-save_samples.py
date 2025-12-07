from modules import ops_file , modulation
from pathlib import Path
Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename_clipped = "logs/barker13_samples_clipped.npy"
filename = "logs/barker13_samples.npy"

samples_clipped = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True )
ops_file.save_samples_2_npfile_v0_1_6 ( filename_clipped , samples_clipped )

samples = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True )
ops_file.save_samples_2_npfile_v0_1_6 ( filename , samples )

