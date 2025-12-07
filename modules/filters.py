import json
import numpy as np
import time as t
import tomllib

from modules import modulation
from numba import njit  # Dodane dla przyspieszenia obliczeń
from numpy.typing import NDArray
from scipy.signal import lfilter , upfirdn

with open ( "settings.json" , "r" ) as settings_json_file :
    settings = json.load ( settings_json_file )
    filter = settings[ "rrc_filter" ]

with open ( "settings.toml" , "rb" ) as settings_toml_file :
    toml_settings = tomllib.load ( settings_toml_file )

BETA = float ( toml_settings[ "rrc_filter" ][ "BETA" ] )
SPAN = int ( toml_settings[ "rrc_filter" ][ "SPAN" ] )

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
    h[ np.isnan ( h ) | np.isinf ( h ) ] = 0
    h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja

    return h

def apply_tx_rrc_filter_v0_1_6 ( symbols: NDArray[ np.complex128 ] , upsample: bool = True , ) -> NDArray[ np.complex128 ] :
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
    rrc_taps = rrc_filter_v4 ( BETA , modulation.SPS , SPAN )
    
    if upsample:
        filtered = upfirdn ( rrc_taps , symbols , modulation.SPS )  # Auto-upsampling + filtracja
    else:
        filtered = lfilter ( rrc_taps , 1.0 , symbols )     # Tylko filtracja
    
    return ( filtered + 0j ) .astype ( np.complex128 )

def apply_tx_rrc_filter_v0_1_5 ( symbols: np.complex128 , upsample: bool = True , ) -> np.complex128 :
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
    rrc_taps = rrc_filter_v4 ( BETA , modulation.SPS , SPAN )
    
    if upsample:
        filtered = upfirdn ( rrc_taps , symbols , modulation.SPS )  # Auto-upsampling + filtracja
    else:
        filtered = lfilter ( rrc_taps , 1.0 , symbols )     # Tylko filtracja
    
    return ( filtered + 0j ) .astype ( np.complex128 )  

def apply_tx_rrc_filter_v0_1_3 ( symbols: np.ndarray , upsample: bool = True , ) -> np.ndarray :
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
    rrc_taps = rrc_filter_v4 ( BETA , modulation.SPS , SPAN )
    
    if upsample:
        filtered = upfirdn ( rrc_taps , symbols , modulation.SPS )  # Auto-upsampling + filtracja
    else:
        filtered = lfilter ( rrc_taps , 1.0 , symbols )     # Tylko filtracja
    
    return ( filtered + 0j ) .astype ( np.complex128 )  


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
    
    return ( filtered + 0j ) .astype ( np.complex128 )  

def apply_rrc_rx_filter_v0_1_3 ( rx_samples: NDArray[ np.complex128 ] , downsample: bool = True ) -> NDArray[ np.complex128 ] :
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
    rrc_taps = rrc_filter_v4 ( BETA , modulation.SPS , SPAN )
    
    # Filtracja (uwaga: filtr musi być znormalizowany!)
    filtered = lfilter ( rrc_taps , 1.0 , rx_samples )

    # Downsampling (opcjonalny)
    if downsample :
        filtered = filtered[ : : modulation.SPS ] # Wybierz 1 próbkę na symbol
    
    return filtered.astype ( np.complex128 )  # Gwarancja complex128

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

def has_sync_sequence ( samples , sync_seq ) :
    """
    Detect presence of sync sequence using normalized cross-correlation and a power gate.

    This replaces the previous fragile amplitude-only test which used a hard-coded
    threshold (mean_amplitude > 100) and produced false positives when the
    received samples had different scaling. The new method computes a normalized
    correlation (0..1) and also checks the mean power (dB) in the best-matching
    window to avoid detecting low-energy noise.

    Returns:
        bool
    """
    start_time = t.perf_counter_ns ()
    x = np.asarray(samples)
    tpl = np.asarray(sync_seq)
    n = len(tpl)
    m = len(x)
    if m < n or n == 0:
        return False

    # valid cross-correlation (tpl reversed) -> length m-n+1
    # use complex conjugate on template for proper correlation with complex samples
    corr = np.abs(np.correlate(x, tpl.conj()[::-1], mode='valid'))
    if corr.size == 0:
        return False

    # template energy
    tpl_energy = np.sum(np.abs(tpl) ** 2)

    # rolling window energy for received samples (efficient via cumsum)
    x_sq = np.abs(x) ** 2
    cumsum = np.concatenate(([0.0], np.cumsum(x_sq)))
    window_energy = cumsum[n:] - cumsum[:-n]

    # normalized correlation: corr / (sqrt(E_window * E_template))
    norm_corr = corr / (np.sqrt(window_energy * tpl_energy) + 1e-12)

    peak_idx = int(np.argmax(norm_corr))
    peak_val = float(norm_corr[peak_idx])

    # compute mean power in dB for the best window (helps to reject tiny noise peaks)
    mean_power = window_energy[peak_idx] / float(n)
    mean_power_db = 10 * np.log10(mean_power + 1e-12)

    # Diagnostic print (keeps previous behaviour of printing a diagnostic)
    print(f"is_sync_seq: peak_corr={peak_val:.4f}, mean_power_db={mean_power_db:.2f} dB")

    # Decision thresholds (tunable): require reasonably high normalized correlation
    # and a minimum power level. These defaults are conservative; adjust to taste.
    MIN_CORR = 0.55      # normalized correlation (0..1)
    MIN_POWER_DB = -40.0 # minimum mean power (dB) to accept as signal
    end_time = t.perf_counter_ns ()
    elapsed_ns = end_time - start_time
    print ( f"has_sync_sequence perf: {elapsed_ns/1e6:.3f} ms" )
    return (peak_val >= MIN_CORR) and (mean_power_db >= MIN_POWER_DB)
