from modules import packet, ops_file, modulation
import numpy as np

filename_samples_2_npy = "np.samples/rx_samples_0.1.15_no_samples.npy"
samples_noise = ops_file.open_samples_from_npf(filename_samples_2_npy)
sync_sequence = modulation.generate_barker13_bpsk_samples_v0_1_7(True)

print(f"Testing noise samples: {samples_noise.size} samples")
print(f"Sync sequence size: {sync_sequence.size}")

peaks = packet.detect_sync_sequence_peaks_v0_1_15(samples_noise, sync_sequence, deep=False)
print(f"Peaks found (deep=False): {peaks}")
print(f"Number of peaks: {len(peaks)}")

peaks_deep = packet.detect_sync_sequence_peaks_v0_1_15(samples_noise, sync_sequence, deep=True)
print(f"Peaks found (deep=True): {peaks_deep}")
print(f"Number of peaks: {len(peaks_deep)}")
