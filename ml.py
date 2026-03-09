import numpy as np
from numpy.typing import NDArray
import torch

def iq_to_tensor ( complex_samples : NDArray[ np.complex128 ] , seq_len: int = 256 ) -> torch.Tensor :
    """ Zamienia rwący wektor complex128 na okna Tensor AI: [Batch_Size, 2_Kanały, Długość_Ramki] """
    # 1. Pocięcie strumienia z radia na równe "klatki"
    num_frames = len ( complex_samples ) // seq_len
    truncated = complex_samples[ : num_frames * seq_len ] # Ucinamy resztkę
    frames = truncated.reshape ( num_frames , seq_len )
    
    # 2. Rozbicie na I (Real) oraz Q (Imag) i rzutowanie na wymuszone float32
    i_chan = np.real ( frames ).astype ( np.float32 )
    q_chan = np.imag ( frames ).astype ( np.float32 )
    
    # 3. Złożenie w tensor [Ilość_Klatek, 2_Kanały, Okno_Czasowe]
    iq_tensor = np.stack ( (i_chan , q_chan ) , axis = 1 )
    
    # 4. Normalizacja (Krytyczne! AI nienawidzi dużych liczb, oczekuje ich w okolicach [-1.0, 1.0])
    # Używamy małej stałej 1e-9 by uniknąć dzielenia przez zero przy pustym szumie
    iq_tensor = iq_tensor / ( np.max ( np.abs ( iq_tensor ) ) + 1e-9 )
    
    return torch.from_numpy ( iq_tensor )