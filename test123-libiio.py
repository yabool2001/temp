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

for channel in phy.channels :
        try:
            print ( channel['attrs'].label.value )
        except OSError:
            value = "N/A (OSError)"
        print(f"{value=} {channel.id=}")
pass