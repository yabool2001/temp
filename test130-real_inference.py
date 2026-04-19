import os , torch , numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Zaciągamy moduł, do którego przeniosłeś architekturę i ładowarkę
from modules import ml , plot

script_filename = os.path.basename ( __file__ )
np.set_printoptions ( threshold = 10 , edgeitems = 3 ) # Ogranicza renderowanie podglądu dużych tablic dla debuggera do ułamka sekundy

device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
print ( f"🔥 {device=}" )


def demoduluj_na_zywo ( sciezka_do_pliku_npy: str ) -> np.ndarray :
    """
    Czysta funkcja produkcyjna. Zero krojowni i datasetów z PyTorcha.
    Wchodzi szum -> Wychodzi zdemodulowany sygnał.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inicjalizacja produkcyjnego silnika AI na: {device}")

    # 1. WSKRZESZAMY MÓZG (W produkcji ten krok robisz raz przy starcie aplikacji)
    model = ml.HardcoreComplexEqualizer().to(device)
    model.load_state_dict(torch.load("bpsk_modem_002.pth", map_location=device, weights_only=True))
    
    # KRYTYCZNE: Blokujemy sieć do odczytu (tryb ewaluacji). 
    # Bez tego system próbowałby liczyć gradienty i wywaliłby pamięć RAM w kosmos.
    model.eval()

    # 2. NASŁUCH Z ANTENY (Czytamy tylko plik wejściowy X!)
    print(f"📡 Wczytuję surowy eter z: {sciezka_do_pliku_npy}")
    surowy_sygnal = np.load(sciezka_do_pliku_npy).astype(np.complex64)
    
    # 3. KROJENIE W LOCIE (Przygotowanie matrycy pod kartę)
    rozmiar_okna = ml.CHUNK_SAMPLES_LEN
    ile_klatek = len(surowy_sygnal) // rozmiar_okna
    
    # Obcinamy ewentualny ogon, jeśli plik nie jest idealną wielokrotnością 8192
    sygnal_obciety = surowy_sygnal[:ile_klatek * rozmiar_okna]
    
    # Magia NumPy (Zero-Copy): Z płaskiego 1D [np. 573440] robimy trójwymiarową 
    # matrycę dla sieci: [70 klatek, 1 kanał, 8192 sampli]. Błyskawicznie!
    x_matryca = sygnal_obciety.reshape((ile_klatek, 1, rozmiar_okna))
    x_tensor = torch.from_numpy(x_matryca).to(device)
    
    # CYFROWE AGC (Automatyczna Kontrola Wzmocnienia)
    # Znajdujemy szczyt amplitudy dla każdej z 70 klatek osobno i normalizujemy,
    # dokładnie tak samo, jak to robiliśmy podczas treningu!
    max_vals = torch.max(torch.abs(x_tensor), dim=2, keepdim=True)[0] + 1e-9
    x_tensor = x_tensor / max_vals

    # 4. 🔥 STRZAŁ Z AI 🔥
    print(f"🧠 RTX 5080 połyka wszystkie {ile_klatek} klatek naraz i odkręca fazę...")
    with torch.no_grad():
        predykcja_ai = model(x_tensor) 

    # 5. SPŁASZCZANIE Z POWROTEM DO 1D (Format do Plotly/Dekodera)
    # AI wypluło matrycę [70, 8192]. Rozwijamy ją w jeden długi sznurek:
    zdemodulowany_sygnal = predykcja_ai.cpu().numpy().flatten().astype(np.complex128)
    
    print(f"✅ Demodulacja zakończona! Zwrócono strumień długości {zdemodulowany_sygnal.size}")
    return zdemodulowany_sygnal

# ==========================================
# UŻYCIE W TWOIM PROGRAMIE GŁÓWNYM
# ==========================================
if __name__ == "__main__":
    
    # Bierzemy obojętnie który plik z radia, byle bez odpowiedzi!
    PLIK_RX = "np.tensors_002_inference/1776615893939_rx_samples.npy"
    
    # Magia dzieje się w 1 linijce:
    odzyskane_symbole = demoduluj_na_zywo(PLIK_RX)
    
    # Otrzymujesz na tacy "odzyskane_symbole". To jest właśnie Twój wygenerowany 
    # przez sztuczną inteligencję odpowiednik "sig_target" z poprzedniego pliku!
    
    # Żeby wyrysować linię z odzyskanymi prostokątami/falą w Plotly:
    # px.line(y=odzyskane_symbole[:5000].real, title="Wyczyszczone Symbole z AI")
    
    # A jeśli potrzebujesz twardych bitów 0 i 1 by wrzucić je do parsowania pakietów:
    bity = (odzyskane_symbole.real > 0).astype(int)
    print("\nFragment zdekodowanego strumienia bitów:")
    print(bity[:100])

plot.complex_waveform_v0_1_6 ( odzyskane_symbole , f"{odzyskane_symbole.size=}" )
#plot.plot_bpsk_symbols_v2 ( odzyskane_symbole , f"{odzyskane_symbole.size=}" )
plot.real_waveform_v0_1_6 ( bity , f"{bity.size=}" )
