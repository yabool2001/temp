import adi

TX_GAIN = -10
RX_GAIN = 70
#GAIN_CONTROL = "slow_attack"
GAIN_CONTROL = "fast_attack"
#GAIN_CONTROL = "manual"
NUM_SAMPLES = 100e3


def init_pluto ( uri , f_c , f_s ) :
    sdr = adi.Pluto ( uri )
    sdr.rx_lo = int ( f_c )
    sdr.sample_rate = int ( f_s )
    #sdr.rx_rf_bandwidth = int ( BW )
    sdr.rx_buffer_size = int ( NUM_SAMPLES )
    sdr.tx_hardwaregain_chan0 = float ( TX_GAIN )
    sdr.gain_control_mode_chan0 = GAIN_CONTROL
    sdr.rx_hardwaregain_chan0 = RX_GAIN
    sdr.rx_output_type = "SI"
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    return sdr


def tx_cyclic ( samples , sdr ) :
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    sdr.tx_cyclic_buffer = True
    sdr.tx ( samples )

def stop_tx_cyclic ( sdr ) :
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    print ( f"{sdr.tx_cyclic_buffer=}" )

def rx_samples ( sdr ) :
    return sdr.rx ()