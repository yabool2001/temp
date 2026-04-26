# Dziennik projektu

## 2026.04.26

*   **`np.tensors_004`** zawiera 3 oryginalne ramki wygenerowane przez `test128-tx_simple_frame.py` i zapisane przez `test128-rx_simple_frane.py`. Rami były transmitowane przez radia umieszczone w tym samym pokoju. tx transmitował z `gain = -10.0`, a rx odbierał w trybie `slow_attack`.


## 2026.04.25

*   **`np.tensors`** zawiera 20 oryginalnych ramek wygenerowanych przez `test126-tx_large_data.py` i zapisanych przez `test126-rx_large_data.py`. Rami były transmitowane przez radia umieszczone w oddzielnych pokojach. tx transmitował z `gain = 0.0`, a rx odbierał w trybie `slow_attack`.
*   **`np.tensors_003`** zawiera kopię `np.tensors`
*   **`np.tensors_003_inference`** zawiera 20 zaagregowanych plików z 20 ramkami zapisanych przez skrypt `test127-rx_tensors_from_all_rx_samples_in_dir.py`
