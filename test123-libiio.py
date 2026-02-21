'''
ad9361-phy

warstwa RF/PHY transceivera AD9361
tu masz głównie konfigurację i telemetrię: LO, gain, BW, rate, RSSI, kalibracje itp.
to nie jest główne źródło strumienia IQ do bufora

cf-ad9361-lpc

RX data path w FPGA (capture core)
daje kanały IQ odbiornika do bufora IIO (czyli właściwe próbki RX)

cf-ad9361-dds-core-lpc

TX generator DDS w FPGA
służy do generowania tonów/testowego sygnału na nadajnik (parametry częstotliwości, skali, fazy), a nie do pomiaru RX

one-bit-adc-dac

pomocniczy 1-bitowy interfejs GPIO/sterowanie (debug/trigger/proste I/O)
wartości binarne lub proste stany, nie pełne IQ

xadc (Xilinx ADC) - monitor analogowy Xilinxa (FPGA)
- temperatura i napięcia zasilania (health monitoring), nie sygnał radiowy
Jeśli nie zajmujesz się diagnostyką sprzętową (np. nie piszesz sterownika zarządzania energią),
możesz te kanały (voltage4 i wyższe w XADC) całkowicie ignorować w swoim programie radiowym.

W skrócie: ad9361-phy = sterowanie radiem, cf-ad9361-lpc = próbki RX IQ, cf-ad9361-dds-core-lpc = generator TX, one-bit-adc-dac = 1-bit I/O, xadc = telemetryka sprzętu.

W Twoim pliku voltage4 i kolejne to konkretne szyny zasilające:
voltage4 (label: vccpaux): Napięcie pomocnicze dla logiki programowalnej (Programmable Logic Auxiliary Voltage).


Wartość: raw: 2431 * scale: 0.732... ≈ 1780 mV (1.8V).

voltage5 (label: vccoddr)

Co to jest: Napięcie zasilania pamięci DDR RAM.


Wartość: raw: 1818 * scale: 0.732... ≈ 1331 mV (1.35V).

voltage6 (label: vrefp)

Co to jest: Dodatnie napięcie referencyjne dla samego przetwornika ADC.

voltage7 (label: vrefn)

Co to jest: Ujemne napięcie referencyjne (masa analogowa).
'''
import iio
import tomllib
import sys
from modules import sdr

with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

BW = int ( toml_settings["ADALM-Pluto"][ "BW" ] )
F_S = int ( BW * 3 if ( BW * 3 ) >= 521100 and ( BW * 3 ) <= 61440000 else 521100 )

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

if pluto_ip is None and pluto_usb is None:
    print ( "There is no Pluto connected (neither USB nor IP). Exiting script." )
    raise SystemExit ( 0 )

ctx = iio.Context ( pluto_usb ) if uri_preference == "usb" else iio.Context ( pluto_ip )

phy = None

for dev in ctx.devices:
    if dev.name and "-phy" in dev.name : phy = dev
    if dev.channels:
        for chan in dev.channels:
            print("{} - {}".format(dev.name, chan._id))
    else:
        print("{}".format(dev.name))

if phy is None:
    print ( "No ad9361-phy found in IIO context. Device not connected — exiting script without traceback." )
    raise SystemExit ( 0 )

for channel in phy.channels :
    try:
        print ( f"{channel.id=} - {channel.attrs['label'].value=}" )
    except ( KeyError , OSError ):
        print ( f"{channel.id=} - label=N/A" )
    if channel.id == "voltage0" and "sampling_frequency" in channel.attrs:
        side = "TX" if channel.output else "RX"
        print(f"{side}: {channel.attrs['sampling_frequency'].value=}")

for channel in phy.channels:
    if channel.id == "voltage0" and "sampling_frequency" in channel.attrs:
        channel.attrs[ "sampling_frequency" ].value = str ( int ( sdr.F_S ) )

for channel in phy.channels:
    if channel.id == "voltage0" and "rf_bandwidth" in channel.attrs:
        channel.attrs["rf_bandwidth"].value = str ( int ( sdr.BW ) )

for channel in phy.channels:
    if channel.id.startswith("altvoltage") and "frequency" in channel.attrs:
        channel.attrs[ "frequency" ].value = str ( int ( sdr.F_C ) )

for channel in phy.channels:
    if channel.id == toml_settings["ADALM-Pluto"]["channels"]["rx0tx0_channel_id"] and "sampling_frequency" in channel.attrs:
        print ( f"{channel.id=} - {channel.output=} {int ( channel.attrs[ 'sampling_frequency' ].value )=:,} Hz" )

for channel in phy.channels:
    if channel.id == toml_settings["ADALM-Pluto"]["channels"]["rx0tx0_channel_id"] and "rf_bandwidth" in channel.attrs:
        side = "TX" if channel.output else "RX"
        print ( f"{side} rf_bandwidth ustawione na {int ( channel.attrs['rf_bandwidth'].value ):,} Hz" )

for channel in phy.channels:
    if channel.id == toml_settings["ADALM-Pluto"]["channels"]["rx0tx0_channel_id"] and channel.output == toml_settings["ADALM-Pluto"]["channels"]["rx_channel_output"] :
        print ( f"{channel.id=} {channel.output=} {int ( channel.attrs[ 'sampling_frequency' ].value )=:,} Hz" )
        sdr.f_s_rx0_readback = int ( channel.attrs[ "sampling_frequency" ].value )
        print ( f"{sdr.f_s_rx0_readback=:,} Hz" )

for channel in phy.channels:
    if channel.id == toml_settings["ADALM-Pluto"]["channels"]["rx0tx0_channel_id"] and channel.output == toml_settings["ADALM-Pluto"]["channels"]["tx_channel_output"] :
        print ( f"{channel.id=} {channel.output=} {int ( channel.attrs[ 'sampling_frequency' ].value )=:,} Hz" )
        sdr.f_s_tx0_readback = int ( channel.attrs[ "sampling_frequency" ].value )
        print ( f"{sdr.f_s_tx0_readback=:,} Hz" )

for channel in phy.channels:
    if channel.id == toml_settings["ADALM-Pluto"]["channels"]["rx0tx0_channel_id"] and channel.output == toml_settings["ADALM-Pluto"]["channels"]["rx_channel_output"] :
        print ( f"{channel.id=} {channel.output=} {int ( channel.attrs[ 'rf_bandwidth' ].value )=:,} Hz" )
        sdr.bw_rx0_readback = int ( channel.attrs[ 'rf_bandwidth' ].value )
        print ( f"{sdr.bw_rx0_readback=:,} Hz" )

for channel in phy.channels:
    if channel.id == toml_settings[ 'ADALM-Pluto' ][ 'channels' ][ 'rx0tx0_channel_id' ] and channel.output == toml_settings[ 'ADALM-Pluto' ][ 'channels' ][ 'tx_channel_output' ] :
        hardwaregain_db = float ( channel.attrs[ 'hardwaregain' ].value.split()[0] )
        print ( f"{channel.id=} {channel.output=} {hardwaregain_db=:.2f} dB" )
        sdr.bw_tx0_gain = hardwaregain_db
        print ( f"{sdr.bw_tx0_gain=:.2f} dB" )

for channel in phy.channels:
    if channel.id == toml_settings[ 'ADALM-Pluto' ][ 'channels' ][ 'rx0tx0_channel_id' ] and channel.output == toml_settings[ 'ADALM-Pluto' ][ 'channels' ][ 'tx_channel_output' ] :
        print ( f"{channel.id=} {channel.output=} {int ( channel.attrs[ 'rf_bandwidth' ].value )=:,} Hz" )
        sdr.bw_tx0_readback = int ( channel.attrs[ 'rf_bandwidth' ].value )
        print ( f"{sdr.bw_tx0_readback=:,} Hz" )

for channel in phy.channels:
    if channel.id == toml_settings["ADALM-Pluto"]["channels"]["lo_rx0_channel_id"] and channel.name == toml_settings["ADALM-Pluto"]["channels"]["lo_rx0_channel_name"] :
        print ( f"{channel.id=} {channel.name=} {int ( channel.attrs[ 'frequency' ].value )=:,} Hz" )
        sdr.f_c_rx0_readback = int ( channel.attrs[ "frequency" ].value )
        print ( f"{sdr.f_c_rx0_readback=:,} Hz" )

for channel in phy.channels:
    if channel.id == toml_settings["ADALM-Pluto"]["channels"]["lo_tx0_channel_id"] and channel.name == toml_settings["ADALM-Pluto"]["channels"]["lo_tx0_channel_name"] :
        print ( f"{channel.id=} {channel.name=} {int ( channel.attrs[ 'frequency' ].value )=:,} Hz" )
        sdr.f_c_tx0_readback = int ( channel.attrs[ "frequency" ].value )
        print ( f"{sdr.f_c_tx0_readback=:,} Hz" )


# gain_control_mode.value = "manual"
# hardwaregain.value = 0.0