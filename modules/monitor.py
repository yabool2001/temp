import numpy as np
import matplotlib.pyplot as plt

def plot_fft ( samples , f_s ) :
    psd = np.fft.fftshift ( np.abs ( np.fft.fft ( samples ) ) )
    f = np.linspace ( -f_s / 2.0 , f_s / 2.0 , len ( psd ) )
    plt
    plt.plot ( f , psd )
    plt.show ()

def plot_fft_p2 ( samples , f_s ) :
    samples = samples**2
    psd = np.fft.fftshift ( np.abs ( np.fft.fft ( samples ) ) )
    f = np.linspace ( -f_s / 2.0 , f_s / 2.0 , len ( psd ) )
    plt
    plt.plot ( f , psd )
    plt.show ()
