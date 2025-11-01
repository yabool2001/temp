Prerequisites:

# To support CUDA with Python ≥ 3.14.0, you need to run:
python -m pip install numba==0.63.0b1
Alternatively, check for a newer version at: https://pypi.org/project/numba/#history

AD9361 block diagram
![image](https://github.com/user-attachments/assets/aa3e2089-f667-406d-b144-5c89a048f7e0)

Key functions:
1. Correlates the received signal with a reference waveform (a modulated Barker 13 sequence) and estimates the phase offset (θ) required for constellation rotation correction


If you want to analyse saved samples there are 2 sets:
1. `logs/rx_samples_10k.csv` containg 10 000 samples
1. `logs/rx_samples_32768.csv` containg 32 768 samples
Sets contain result of cyclic transmitt of packet saved in `logs/tx_samples_393.csv`. If you want to load this data to samples use `ops_file.open_csv_and_load_np_complex128 ( filename )` function.
F_C=820000000.0, F_S=3000000, BW=1000000, SPS=4, RRC_BETA=0.35, RRC_SPAN=11.
BARKER13_W_PADDING=[6, 80], length_byte=[3], payload=[15, 15, 15, 15], crc32_bytes=[101, 92, 137, 41]
packet_bits=array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
       1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
       0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
      dtype=uint8)
tx_bpsk_symbols=array([-1, -1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1,
       -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1,
       -1,  1,  1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1,
        1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,
       -1, -1,  1])