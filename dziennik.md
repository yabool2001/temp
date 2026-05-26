# Dziennik projektu

## 2026.05.26 testy nowej funkcji clip_samples_and_create_tensor_4_training któ©a bazuje na wklejaniu sampli a nie symboli i jest wielki progres
Epoka [01/15] | Błąd EVM (MSE): 0.12287 | Czas epoki: 114.12 s
Epoka [02/15] | Błąd EVM (MSE): 0.11848 | Czas epoki: 113.07 s
Epoka [03/15] | Błąd EVM (MSE): 0.11810 | Czas epoki: 115.24 s
Epoka [04/15] | Błąd EVM (MSE): 0.11819 | Czas epoki: 110.82 s
Epoka [05/15] | Błąd EVM (MSE): 0.11821 | Czas epoki: 110.80 s
Epoka [06/15] | Błąd EVM (MSE): 0.11806 | Czas epoki: 110.98 s
Epoka [07/15] | Błąd EVM (MSE): 0.11818 | Czas epoki: 110.55 s
Epoka [08/15] | Błąd EVM (MSE): 0.11817 | Czas epoki: 110.78 s
Epoka [09/15] | Błąd EVM (MSE): 0.11806 | Czas epoki: 110.65 s
Epoka [10/15] | Błąd EVM (MSE): 0.11795 | Czas epoki: 110.76 s
Epoka [11/15] | Błąd EVM (MSE): 0.11811 | Czas epoki: 112.09 s
Epoka [12/15] | Błąd EVM (MSE): 0.11797 | Czas epoki: 112.40 s
Epoka [13/15] | Błąd EVM (MSE): 0.11797 | Czas epoki: 113.66 s

## 2026.05.20 praca w Lipkowie
*   **`np.tensors_004`** zawiera 2 x 8 oryginalnych sampli wygenerowane przez `test134-tx_large_frames.py` i zapisanych przez `test134-rx_large_frames.py`. Sample były transmitowane przez radia połączone tłumikiem 30 dB. tx transmitował z `gain = -10.0`, a rx odbierał w trybie `slow_attack`.
