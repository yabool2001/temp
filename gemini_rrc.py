import numpy as np
from scipy import signal
from scipy.signal import windows # Importujemy windows

# https://stackoverflow.com/questions/14614966/easy-way-to-implement-a-root-raised-cosine-rrc-filter-using-python-numpy
def rrc_filter(samples: np.ndarray, sps: int, beta: float, span: int) -> np.ndarray:
    """
    Filtruje próbki BPSK przy użyciu filtru Root Raised Cosine (RRC).

    Args:
        samples (np.ndarray): Próbki BPSK (complex128) do filtrowania.
                              Otrzymywane z rx() lub przygotowywane do tx() z PyADI.
        sps (int): Próbki na symbol (Samples Per Symbol).
        beta (float): Współczynnik roll-off filtru RRC (0.0 do 1.0).
        span (int): Długość filtru w symbolach. Im większy span, tym dłuższy filtr
                    i lepsza charakterystyka, ale większe opóźnienie i koszt obliczeniowy.

    Returns:
        np.ndarray: Przefiltrowane próbki (complex128).
    """

    if not isinstance(samples, np.ndarray) or samples.dtype != np.complex128:
        raise TypeError("Argument 'samples' musi być tablicą numpy typu complex128.")
    if not isinstance(sps, int) or sps <= 0:
        raise ValueError("Argument 'sps' musi być dodatnią liczbą całkowitą.")
    if not isinstance(beta, (int, float)) or not (0.0 <= beta <= 1.0):
        raise ValueError("Argument 'beta' musi być liczbą zmiennoprzecinkową z zakresu [0.0, 1.0].")
    if not isinstance(span, int) or span <= 0:
        raise ValueError("Argument 'span' musi być dodatnią liczbą całkowitą.")

    # Generowanie impulsowej odpowiedzi filtru RRC
    # Prawidłowa funkcja to scipy.signal.windows.rrc
    num_taps = sps * span + 1  # Całkowita liczba kranów (próbek) filtru

    # scipy.signal.windows.rrc przyjmuje num_taps, beta i sps
    h_rrc = windows.rrc(num_taps, beta=beta, sps=sps)

    # Konwolucja próbek z filtrem RRC
    filtered_samples = np.convolve(samples, h_rrc, mode='full')

    # Konwolucja może zmienić typ danych na complex64, jeśli h_rrc jest float64.
    # Upewniamy się, że wynik jest complex128, tak jak wymagane przez PyADI.
    return filtered_samples.astype(np.complex128)

# Przykładowe użycie funkcji pozostaje bez zmian i powinno teraz działać poprawnie.
if __name__ == "__main__":
    # Symulacja próbek BPSK (complex128)
    num_symbols = 100
    sps_example = 8
    beta_example = 0.35
    span_example = 4

    random_bits = np.random.randint(0, 2, num_symbols)
    bpsk_symbols = 2 * random_bits - 1

    upsampled_samples_tx = np.zeros(num_symbols * sps_example, dtype=np.complex128)
    upsampled_samples_tx[::sps_example] = bpsk_symbols.astype(np.complex128)

    print(f"Liczba próbek przed filtrem (do TX): {len(upsampled_samples_tx)}")

    filtered_tx_samples = rrc_filter(upsampled_samples_tx, sps_example, beta_example, span_example)
    print(f"Liczba próbek po filtrowaniu (do TX): {len(filtered_tx_samples)}")
    print(f"Typ danych próbek TX po filtrowaniu: {filtered_tx_samples.dtype}")

    noise = (np.random.randn(len(filtered_tx_samples)) + 1j * np.random.randn(len(filtered_tx_samples))) * 0.1
    received_samples = filtered_tx_samples + noise
    received_samples = received_samples.astype(np.complex128)

    print(f"\nLiczba odebranych próbek (z rx()): {len(received_samples)}")

    filtered_rx_samples = rrc_filter(received_samples, sps_example, beta_example, span_example)
    print(f"Liczba próbek po filtrowaniu (z RX): {len(filtered_rx_samples)}")
    print(f"Typ danych próbek RX po filtrowaniu: {filtered_rx_samples.dtype}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(np.real(upsampled_samples_tx), label='Próbki BPSK przed filtrem (TX) - część rzeczywista')
        plt.plot(np.imag(upsampled_samples_tx), label='Próbki BPSK przed filtrem (TX) - część urojona')
        plt.title('Próbki BPSK przed filtrem RRC (do TX)')
        plt.xlabel('Numer próbki')
        plt.ylabel('Amplituda')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(np.real(filtered_tx_samples), label='Próbki BPSK po filtrze RRC (do TX) - część rzeczywista')
        plt.plot(np.imag(filtered_tx_samples), label='Próbki BPSK po filtrowaniu (do TX) - część urojona')
        plt.title('Przefiltrowane próbki BPSK po filtrowaniu RRC (do TX)')
        plt.xlabel('Numer próbki')
        plt.ylabel('Amplituda')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(np.real(received_samples), label='Odebrane próbki (z szumem) - część rzeczywista')
        plt.plot(np.imag(received_samples), label='Odebrane próbki (z szumem) - część urojona')
        plt.title('Odebrane próbki (z symulowanego rx() PyADI)')
        plt.xlabel('Numer próbki')
        plt.ylabel('Amplituda')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(np.real(filtered_rx_samples), label='Przefiltrowane odebrane próbki - część rzeczywista')
        plt.plot(np.imag(filtered_rx_samples), label='Przefiltrowane odebrane próbki - część urojona')
        plt.title('Przefiltrowane odebrane próbki (po filtrowaniu RRC RX)')
        plt.xlabel('Numer próbki')
        plt.ylabel('Amplituda')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nZainstaluj matplotlib (pip install matplotlib) aby zobaczyć wykresy.")