import numpy as np
from scipy.signal import lfilter

def rrc_filter ( beta , sps , num_taps ):
    N = num_taps * sps
    t = np.arange ( -N // 2 , N // 2 + 1 ) / sps
    taps = np.zeros_like ( t )

    for i in range ( len ( t ) ):
        if t[i] == 0.0:
            taps[i] = 1.0 - beta + ( 4 * beta / np.pi )
        elif abs ( t[i] ) == 1 / ( 4 * beta ):
            taps[i] = ( beta / np.sqrt ( 2 ) ) * (
                ( 1 + 2 / np.pi ) * np.sin ( np.pi / ( 4 * beta ) ) +
                ( 1 - 2 / np.pi ) * np.cos ( np.pi / ( 4 * beta ) )
            )
        else:
            numerator = (
                np.sin ( np.pi * t[i] * ( 1 - beta ) ) +
                4 * beta * t[i] * np.cos ( np.pi * t[i] * ( 1 + beta ) )
            )
            denominator = (
                np.pi * t[i] * ( 1 - ( 4 * beta * t[i] ) ** 2 )
            )
            taps[i] = numerator / denominator

    return taps / np.sqrt ( np.sum ( taps ** 2 ) )

def rrc_filter(beta, sps, span):
    """
    Generuje współczynniki filtra RRC.
    
    :param beta: roll-off factor (np. 0.35)
    :param sps: samples per symbol (np. 8)
    :param span: długość filtra w symbolach (np. 11)
    :return: współczynniki filtra (1D numpy array)
    """
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif beta != 0 and np.abs(t[i]) == 1 / (4 * beta):
            h[i] = (beta / np.sqrt(2)) * \
                   ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + 
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            numerator = np.sin(np.pi * t[i] * (1 - beta)) + \
                        4 * beta * t[i] * np.cos(np.pi * t[i] * (1 + beta))
            denominator = np.pi * t[i] * (1 - (4 * beta * t[i]) ** 2)
            h[i] = numerator / denominator

    h = h / np.sqrt(np.sum(h ** 2))  # normalizacja energetyczna
    return h

def apply_rrc_filter(rx_samples, beta=0.35, sps=4, span=11):
    """
    Filtruje dane wejściowe za pomocą filtra RRC.
    
    :param rx_samples: próbki zespolone (np.ndarray, complex)
    :return: przefiltrowane próbki
    """
    rrc_taps = rrc_filter_v2(beta, sps, span)
    filtered = lfilter(rrc_taps, 1.0, rx_samples)
    return filtered


def rrc_filter_v2 ( sps , beta , span ) :
    N = sps * span
    t = np.arange(-N//2, N//2 + 1, dtype=np.float64) / sps
    taps = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
    taps[np.isnan(taps)] = 0
    taps /= np.sqrt(np.sum(taps**2))
    return taps