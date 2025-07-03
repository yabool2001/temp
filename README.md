AD9361 block diagram
![image](https://github.com/user-attachments/assets/aa3e2089-f667-406d-b144-5c89a048f7e0)

Key functions:
1. Correlates the received signal with a reference waveform (a modulated Barker 13 sequence) and estimates the phase offset (θ) required for constellation rotation correction


If you want to analyse saved samples there are 2 sets:
1. `rx_samples_10k.csv` containg 10 000 samples
1. `rx_samples_32768.csv` containg 32 768 samples
Sets contain result of cyclic transmitt of packet saved in `tx_samples_393.csv`
