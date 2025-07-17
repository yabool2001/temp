
import adi
import csv
import iio
import numpy as np
import os
import time as t

filename = "logs/rx_samples_test103_raw_data.csv"

script_filename = os.path.basename ( __file__ )

contexts = iio.scan_contexts ()
usb_match = None
for uri, description in contexts.items():
    if "10447318ac0f00091e002400454e18b77d" in description:
        uri_tx = uri
sdr = adi.Pluto ( uri_tx )
sdr.rx_lo = int ( 2400000000 )
sdr.rx_rf_bandwidth = int ( 4000000 )
sdr.gain_control_mode_chan0 = "slow_attack"
sdr.rx_output_type = "SI"
t.sleep ( 0.2 ) #delay after setting device parameters
samples = sdr.rx()
print ( "Max scaled value:", np.max ( np.abs ( samples ) ) )
with open ( filename , mode = 'w' , newline = '') as file :
    writer = csv.writer ( file )
    writer.writerow ( [ 'real' , 'imag' ] )  # nagłówki kolumn
    for sample in samples :
        writer.writerow ( [ sample.real , sample.imag ] )
