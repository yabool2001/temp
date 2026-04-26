# Dziennik projektu

## 2026.04.25

*   **`np.tensors`** zawierają 20 oryginalnych ramek wygenerowanych przez `test126-tx_large_data.py` i zapisanych przez `test126-rx_large_data.py`. Rami były transmitowane przez radia umieszczone w oddzielnych pokojach. tx transmitował z `gain = 0.0`, a rx odbierał w trybie `slow_attack`.
*   **`np.tensors_003`** zawierają kopię `np.tensors`
*   **`np.tensors_003_inference`** zawiera 20 zaagregowanych plików z 20 ramkami zapisanych przez skrypt `test127-rx_tensors_from_all_rx_samples_in_dir.py`
