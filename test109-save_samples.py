from modules import ops_file , plot , modulation
from pathlib import Path

Path ( "logs" ).mkdir ( parents = True , exist_ok = True )

filename_clipped = "np.samples/barker13_samples_clipped.npy"
filename = "np.samples/barker13_samples.npy"

barker13 = modulation.get_barker13_bpsk_samples_v0_1_3 ()
barker13_clipped = modulation.get_barker13_bpsk_samples_v0_1_3 ( clipped = True )


plot.complex_waveform_v0_1_6 ( barker13 , f"{barker13.size=}" , False )
plot.complex_waveform_v0_1_6 ( barker13_clipped , f"{barker13_clipped.size=}" , False )

#ops_file.save_samples_2_npf ( filename , barker13 )
#ops_file.save_samples_2_npf ( filename_clipped , barker13_clipped )
