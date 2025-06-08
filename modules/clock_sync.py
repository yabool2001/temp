# modules/clock_sync.py

import numpy as np
from scipy.signal import lfilter

def polyphase_clock_sync(samples, sps, nfilts, excess_bw):
    """
    Prosta symulacja działania filtru synchronizacji zegara metodą polyphase.
    Uwaga: To NIE jest prawdziwy blok PFB z GNU Radio, tylko przybliżenie.

    :param samples: Zespolone próbki wejściowe (numpy array)
    :param sps: Samples per symbol
    :param nfilts: Liczba podfiltrów (standardowo 32)
    :param excess_bw: Nadmiarowe pasmo RRC (np. 0.35)
    :return: Przybliżona wersja próbek po synchronizacji zegara
    """
    from scipy.signal import firwin

    # Użycie prostego RRC jako przybliżenia PFB taps
    num_taps = 11 * sps * nfilts
    beta = excess_bw
    # Zastąpienie firdes.root_raised_cosine przybliżeniem
    t = np.arange(-num_taps//2, num_taps//2) / sps
    rrc_taps = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2 + 1e-6)
    rrc_taps /= np.sum(rrc_taps)

    filtered = lfilter(rrc_taps, 1.0, samples)
    # Przybliżone próbkowanie w symbolach
    synced = filtered[::sps]

    return synced
