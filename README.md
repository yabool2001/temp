# Prerequisites:
`pylibiio` `pyadi-iio` `numba` `scipy` `plotly.express`

To support CUDA with newest version of Python, you may need to run newest beta numba, e.g. 0.63.0b1 : `python -m pip install numba==0.63.0c1`
Check for a newest version at: https://pypi.org/project/numba/#history

# Hardware description
## AD9361 transceiver block diagram
![image](https://github.com/user-attachments/assets/aa3e2089-f667-406d-b144-5c89a048f7e0)

# Software description

## Information
All user data is encapsulated in a FRAME that comprise:
- 13 bits of SYNC_SEQUENCE ( e.g. BARKER13)
- 11 bits of PACKET_LEN
- PACKET

PACKET comprise:
- PAYLOAD whose length = PACKET_LEN - 4 bytes of CRC32
- 4 bytes of CRC32

## Samples corrections in corrections module
### CFO estimation
Implementacja wykonana na podstawie opisu w rozdz.10.5 CFO Estimation książki "Software-Deﬁned Radio for Engineers" Travis F. Collins, Robin Getz, Di Pu, Alexander M. Wyglinski

1. Znajdź korelację, wyznacz pozycję preambuły (peak)
1. Wyodrębnij segment sygnału odpowiadający preambule lub powtórzonym blokom preambuły
1. Oblicz średnią fazę iloczynów próbek oddalonych o M próbek (M = separacja powtarzających się symboli lub sps): products = seg[ M: ] * conj ( seg[ : -M ] )
1. Delta_phi = angle ( mean ( products ) )
1. CFO_est_Hz = Delta_phi * fs / ( 2 pi M )
1. Zastosuj korekcję multiplicative: rx = exp ( -j2 pi CFO_est_Hz * ( n / fs ) )
1. Opcjonalnie: uruchom PLL lub fine estimator żeby dopracować residual

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