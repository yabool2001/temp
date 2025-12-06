import numpy as np

from numpy.typing import NDArray

bits_0101 = np.array ( [ 0 , 1 , 0 , 1 ] , dtype = np.uint8 )
print ( bits_0101 )

symbols1 = np.require ( bits_0101 , np.uint8 , ['C'] ) * 2.0 - 1.0 + 0j
print ( symbols1 )


bits = np.require ( bits_0101 , dtype = np.uint8 , requirements = [ 'C' ] )
symbols2 = ( bits * 2.0 - 1.0 ).astype ( np.complex128 )
print ( symbols2 )

# Map 0 -> -1, 1 -> +1
symbols3 = np.where ( bits == 1 , 1.0 + 0j , -1.0 + 0j ).astype ( np.complex128 )
print ( symbols3 )