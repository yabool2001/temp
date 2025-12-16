from modules import corrections , ops_file , plot , modulation
from pathlib import Path
import numpy as np

Path ( "correlation" ).mkdir ( parents = True , exist_ok = True )

plt = True
save = False

filename_sync_sequence_1 = "correlation/sync_sequence_1.npy"
filename_sync_sequence_2 = "correlation/sync_sequence_2.npy"
filename_samples_1 = "correlation/samples_1.npy"
filename_samples_2 = "correlation/samples_2.npy"
filename_samples_2_noisy_1 = "correlation/samples_2_noisy_1.npy"
filename_samples_2_noisy_2 = "correlation/samples_2_noisy_2.npy"
filename_samples_2_noisy_3 = "correlation/samples_2_noisy_3.npy"
filename_samples_2_noisy_4 = "correlation/samples_2_noisy_4.npy"

sync_sequence_1 = np.array ( [ 0 , 100 , 100 , -100 , -100 , 0  ] , dtype = np.int32 )
sync_sequence_2 = np.array ( [ 0 , 100 , 0 , -100 , 0 , 200 , 0 , -200 , 0 , 1000 , 0 , -200 , 0 , 200 , 0 , -100 , 0 , 100 , 0 ] , dtype = np.int32 )

samples_1 = np.array ( [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 
                              0 , 100 , 100 , -100 , -100 , 0  ,
                              0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
                             , dtype = np.int32 )
samples_2 = np.array ( [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 100 , 0 , -100 , 0 , 200 , 0 , -200 , 0 , 1000 , 0 , -200 , 0 , 200 , 0 , -100 , 0 , 100 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
                               , dtype = np.int32 )

samples_2_noisy_1 = np.array ( [    0 , -10 , 0 , 10 , 0 , -10 , 0 , 10 , 0 , -10 ,
                                    0 , -20 , 0 , 20 , 0 , -20 , 0 , 20 , 0 , -20 ,
                                    0 , 10 , 0 , -10 , 0 , 10 , 0 , -10 , 0 , 10 ,
                                    0 , 100 , 0 , -100 , 0 , 200 , 0 , -200 , 0 , 1000 , 0 , -200 , 0 , 200 , 0 , -100 , 0 , 100 , 0 ,
                                    0 , 10 , 0 , -10 , 0 , 10 , 0 , -10 , 0 , 10 ,
                                    0 , -20 , 0 , 20 , 0 , -20 , 0 , 20 , 0 , -20 ,
                                    0 , 10 , 0 , 10 , 0 , 10 , 0 , 10 , 0 , 10 ]
                               , dtype = np.int32 )

# Nowy zaszumiony i jammingowany samples_2_noisy_2 z sync_sequence_2 na tej samej pozycji (indeksy 30-48)
# Dodaję szum (np. ±5-10) do sync_sequence_2 i jamming (np. dodatkowe interferencje)
sync_sequence_2_noisy_jammed = sync_sequence_2 + np.array([0, 5, 0, -8, 0, 12, 0, -15, 0, 20, 0, -10, 0, 8, 0, -5, 0, 3, 0], dtype=np.int32)  # szum
sync_sequence_2_noisy_jammed += np.array([0, 0, 50, 0, -30, 0, 40, 0, -25, 0, 35, 0, -20, 0, 15, 0, -10, 0, 5], dtype=np.int32)  # jamming

samples_2_noisy_2 = np.concatenate([
    np.array([0, -10, 0, 10, 0, -10, 0, 10, 0, -10,
              0, -20, 0, 20, 0, -20, 0, 20, 0, -20,
              0, 10, 0, -10, 0, 10, 0, -10, 0, 10], dtype=np.int32),  # szum przed
    sync_sequence_2_noisy_jammed,  # zaszumiony i jammingowany sync_sequence_2
    np.array([0, 10, 0, -10, 0, 10, 0, -10, 0, 10,
              0, -20, 0, 20, 0, -20, 0, 20, 0, -20,
              0, 10, 0, 10, 0, 10, 0, 10, 0, 10], dtype=np.int32)   # szum po
])

# Szum w zakresie +/-50 lub więcej. Jamming - umieszczenie częściowo podobnych sekwencji w innych miejscach, jak na indeksach 10-20 i 55-65, z połową amplitudy.
samples_2_noisy_3 = np.array([-98, 148, 70, -94, -129, 288, -180, -98, -79, 14,
                              130, -63, 172, -151, 159, -249, -70, -151, 108, 557,
                              143, 93, 185, -9, 76, 360, 113, -179, 52, 35,
                              144, -52, -142, -131, -13, 270, -11, -226, -150, 1163,
                              -146, -157, 119, 130, 106, -166, -180, 228, -34, 73,
                              -213, -112, 115, -187, 41, 64, 95, -148, 235, 139,
                              291, 166, 163, -166, -495, -120, -151, 159, 187, -199,
                              -311, -147, -95, 59, 109, -10, 17, -157, -39], dtype=np.int32)

samples_2_noisy_4 = corrections.generate_noisy_samples ()

if plt :
    #plot.complex_waveform_v0_1_6 ( barker13 , f"{barker13.size=}" , True )
    #plot.complex_waveform_v0_1_6 ( barker13_clipped , f"{barker13_clipped.size=}" , True )
    #plot.real_waveform ( sync_sequence_1 , f"{sync_sequence_1.size=}" , True )
    #plot.real_waveform ( samples_1 , f"{samples_1.size=}" , True )
    #plot.real_waveform ( sync_sequence_2 , f"{sync_sequence_2.size=}" , True )
    #plot.real_waveform ( samples_2 , f"{samples_2.size=}" , True )
    #plot.real_waveform ( samples_2_noisy_1 , f"{samples_2_noisy_1.size=}" , True )
    #plot.real_waveform ( samples_2_noisy_2 , f"{samples_2_noisy_2.size=}" , True )
    #plot.real_waveform ( samples_2_noisy_3 , f"{samples_2_noisy_3.size=}" , True )
    plot.real_waveform ( samples_2_noisy_4 , f"{samples_2_noisy_4.size=}" , True )

if save :
    '''
    #ops_file.save_samples_2_npf ( filename , barker13 )
    #ops_file.save_samples_2_npf ( filename_clipped , barker13_clipped )
    ops_file.save_samples_2_npf ( filename_sync_sequence_1 , sync_sequence_1 )
    ops_file.save_samples_2_npf ( filename_sync_sequence_2 , sync_sequence_2 )
    ops_file.save_samples_2_npf ( filename_samples_1 , samples_1 )
    ops_file.save_samples_2_npf ( filename_samples_2 , samples_2 )
    ops_file.save_samples_2_npf ( filename_samples_2_noisy_1 , samples_2_noisy_1 )
    ops_file.save_samples_2_npf ( filename_samples_2_noisy_2 , samples_2_noisy_2 )
    ops_file.save_samples_2_npf ( "correlation/samples_2_noisy_3.npy" , samples_2_noisy_3 )
    ''' 
    ops_file.save_samples_2_npf ( "correlation/samples_2_noisy_4.npy" , samples_2_noisy_4 )