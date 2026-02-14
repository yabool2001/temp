import iio
import tomllib

with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

uri_preference: str = "usb"

contexts = iio.scan_contexts()

pluto_ip = None
pluto_usb = None

for uri, description in contexts.items():
    if toml_settings["ADALM-Pluto"]["URI"]["SN_RX"] in description:
        if uri.startswith ( "ip:" ) :
            pluto_ip = uri
        elif uri.startswith ( "usb:" ) :
            pluto_usb = uri
ctx = iio.Context ( pluto_usb ) if uri_preference == "usb" else iio.Context ( pluto_ip )

for dev in ctx.devices:
        if "-phy" in dev.name : phy = dev
        if dev.channels:
            for chan in dev.channels:
                print("{} - {}".format(dev.name, chan._id))
        else:
            print("{}".format(dev.name))

for channels_attr in phy.channels :
        try:
            print ( channels_attr._id].value
        except OSError:
            value = "N/A (OSError)"
        print(f"{value=} {channels_attr._id=}")
pass