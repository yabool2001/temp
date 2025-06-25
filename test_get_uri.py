import adi
import iio
from modules import sdr

serial = "1044739a470b000a090018007ecf7f5ea8"
serial = "10447318ac0f00091e002400454e18b77d"

selected_uri = sdr.get_uri ( serial , "ip" )

print ( selected_uri )