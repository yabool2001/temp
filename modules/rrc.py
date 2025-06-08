
### Root Raised Cosine Filter Module ###

import numpy as np

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
