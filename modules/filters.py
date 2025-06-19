import numpy as np
from numba import njit  # Dodane dla przyspieszenia obliczeń
from scipy.signal import lfilter , upfirdn

def rrc_filter_v1 ( beta , sps , num_taps ):
    """
    Generuje współczynniki filtra RRC.
    
    :param beta: roll-off factor (np. 0.35)
    :param sps: samples per symbol (np. 8)
    :param span: długość filtra w symbolach (np. 11)
    :return: współczynniki filtra (1D numpy array)
    """

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

def rrc_filter_v2 ( sps , beta , span ) :
    """
    UWAGA! Ta wersja jest najszybsza (nieznacznie szybsza niż v3), bo używa np.sinc
    ale wymaga beta > 0
    """
    N = sps * span
    t = np.arange ( -N//2 , N//2 + 1 , dtype = np.float64 ) / sps
    taps = np.sinc ( t ) * np.cos ( np.pi * beta * t ) / ( 1 - ( 2 * beta * t ) **2 )
    taps[np.isnan ( taps )] = 0
    taps /= np.sqrt ( np.sum ( taps **2 ) )
    return taps

def rrc_filter_v3 ( beta , sps , span ) :
    """
    DeepSeek -V3 R1
    Filtruje dane wejściowe za pomocą filtra RRC.
    """
    N = span * sps
    t = np.arange ( -N/2 , N/2 + 1 , dtype = np.float64 ) / sps

    # Obsługa beta = 0 (filtr sinc)
    if beta == 0:
        h = np.sinc ( t )
        h = h / np.sqrt ( np.sum ( h**2 ) )  # Normalizacja
        return h
    
    h = np.zeros_like ( t )

    # Przypadek t = 0
    mask_zero = ( t == 0.0 )
    h[mask_zero] = 1.0 - beta + ( 4 * beta / np.pi )

    # Przypadek t = ±1/(4β)
    mask_special = np.abs ( np.abs ( t ) - 1 / ( 4 * beta ) ) < 1e-10  # Tolerancja numeryczna
    h[mask_special] = ( beta / np.sqrt ( 2 ) ) * \
                     ( ( 1 + 2 / np.pi ) * np.sin ( np.pi / ( 4 * beta ) ) + 
                      ( 1 - 2 / np.pi ) * np.cos ( np.pi / ( 4 * beta ) ) )

    # Pozostałe przypadki
    mask_rest = ~( mask_zero | mask_special )
    t_rest = t[mask_rest]
    numerator = np.sin ( np.pi * t_rest * ( 1 - beta ) ) + \
                4 * beta * t_rest * np.cos ( np.pi * t_rest * ( 1 + beta ) )
    denominator = np.pi * t_rest * ( 1 - ( 4 * beta * t_rest ) ** 2 )
    h[mask_rest] = numerator / denominator

    h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja

    return h

import numpy as np


@njit ( cache = True , fastmath = True )  # Kompilacja Just-In-Time z optymalizacjami
def rrc_filter_v4 ( beta , sps , span ) :
    """
    DeepSeek -V3 R1 (Zoptymalizowana)
    Filtruje dane wejściowe za pomocą filtra RRC.
    """
    N = span * sps
    t = np.arange ( -N / 2 , N / 2 + 1 , dtype = np.float64 ) / sps

    # Obsługa beta = 0 (filtr sinc)
    if beta == 0 :
        h = np.sinc ( t )
        h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja
        return h
    
    h = np.zeros_like ( t )

    # Stałe pre-kalkulowane dla wydajności
    beta_pi = np.pi * beta
    inv_4beta = 1 / ( 4 * beta )
    sqrt2 = np.sqrt ( 2 )
    special_val = ( beta / sqrt2 ) * \
                ( ( 1 + 2 / np.pi ) * np.sin ( np.pi / ( 4 * beta ) ) + 
                ( 1 - 2 / np.pi ) * np.cos ( np.pi / ( 4 * beta ) ) )

    # Obliczenia z maskami (zoptymalizowane)
    for i in range ( len ( t ) ) :
        ti = t[i]
        if ti == 0.0 :
            h[i] = 1.0 - beta + ( 4 * beta / np.pi )
        elif np.abs ( np.abs ( ti ) - inv_4beta ) < 1e-10 :  # Tolerancja numeryczna
            h[i] = special_val
        else :
            numerator = np.sin ( np.pi * ti * ( 1 - beta ) ) + \
                      4 * beta * ti * np.cos ( np.pi * ti * ( 1 + beta ) )
            denominator = np.pi * ti * ( 1 - ( 4 * beta * ti ) ** 2 )
            h[i] = numerator / denominator

    # Bezpieczna obsługa NaN/Inf i normalizacja
    h[np.isnan ( h ) | np.isinf ( h )] = 0
    h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja

    return h

def apply_tx_rrc_filter ( symbols: np.ndarray , sps: int = 4 , beta: float = 0.35, span: int = 11 , upsample: bool = True , ) -> np.ndarray :
    """
    Stosuje filtr Root Raised Cosine (RRC) z opcjonalnym upsamplingiem.
    Zawsze zwraca sygnał zespolony (complex128).

    Parametry:
        symbols: Sygnał wejściowy (real lub complex).
        beta: Współczynnik roll-off (0.0-1.0).
        sps: Próbek na symbol (samples per symbol).
        span: Długość filtra w symbolach.
        upsample: Czy wykonać upsampling (True) czy tylko filtrować (False).

    Zwraca:
        Przefiltrowany sygnał zespolony (complex128).
    """
    rrc_taps = rrc_filter_v4 ( beta , sps , span )
    
    if upsample:
        filtered = upfirdn ( rrc_taps , symbols , sps )  # Auto-upsampling + filtracja
    else:
        filtered = lfilter ( rrc_taps , 1.0 , symbols )     # Tylko filtracja
    
    return ( filtered + 0j ) .astype ( np.complex128 )  # Wymuszenie complex128

def apply_rrc_rx_filter ( rx_samples: np.ndarray , sps: int = 4 , beta: float = 0.35 , span: int = 11 , downsample: bool = True ) -> np.ndarray :
    """
    Filtruje odebrane próbki z SDR filtrem RRC.
    
    Parametry:
        rx_samples: Odebrane próbki (complex128) z SDR.
        beta: Współczynnik roll-off filtra.
        sps: Próbek na symbol (musi być zgodny z nadajnikiem!).
        span: Długość filtra w symbolach.
        downsample: Czy zmniejszyć liczbę próbek do 1 na symbol (True/False).
    
    Zwraca:
        Przefiltrowane próbki (complex128).
    """
    # Generuj współczynniki filtra RRC
    rrc_taps = rrc_filter_v4 ( beta , sps , span )
    
    # Filtracja (uwaga: filtr musi być znormalizowany!)
    filtered = lfilter ( rrc_taps , 1.0 , rx_samples )

    # Downsampling (opcjonalny)
    if downsample :
        filtered = filtered[::sps]  # Wybierz 1 próbkę na symbol
    
    return filtered.astype ( np.complex128 )  # Gwarancja complex128