import csv
import pandas as pd
import plotly.express as px

def open_and_init_symbols_csv ( filename ) :
    csv_file = open ( filename , mode = "w" , newline = '' )
    csv_writer = csv.writer ( csv_file )
    csv_writer.writerow ( [ "symbol" ] )
    return csv_file , csv_writer

def open_and_init_complex_samples_csv ( filename ) :
    csv_file = open ( filename , mode = "w" , newline = '' )
    csv_writer = csv.writer ( csv_file )
    csv_writer.writerow ( [ "real" , "imag" ] )
    return csv_file , csv_writer

def open_and_write_symbols_2_csv ( name , symbols ) :
    csv_file , csv_writer = open_and_init_symbols_csv ( name )
    for symbol in symbols :
        csv_writer.writerow ( [ symbol ] )
    return csv_file , csv_writer

def open_and_write_samples_2_csv ( name , samples ) :
    csv_file , csv_writer = open_and_init_complex_samples_csv ( name )
    for sample in samples :
        csv_writer.writerow ( [ sample.real , sample.imag ] )
    return csv_file , csv_writer

def append_symbols_2_csv ( csv_writer , symbols ) :
    for symbol in symbols :
        csv_writer.writerow ( [ symbol ] )
        
def append_samples_2_csv ( csv_writer , samples ) :
    for sample in samples :
        csv_writer.writerow ( [ sample.real , sample.imag ] )

def flush_data ( csv_file ) :
    csv_file.flush ()

def close_csv ( csv_file ) :
    csv_file.close ()

def flush_data_and_close_csv ( csv_file ) :
    flush_data ( csv_file )
    close_csv ( csv_file )

def plot_samples ( filename ) :
    # Wczytanie danych i wyświetlenie wykresu w Plotly
    print ( "Rysuję wykres..." )
    df = pd.read_csv ( filename )
    # Zbuduj sygnał zespolony (opcjonalnie, jeśli chcesz jako 1D)
    # signal = df["real"].values + 1j * df["imag"].values
    # Przygotuj dane do wykresu
    df["index"] = df.index
    # Wykres Plotly Express – wersja liniowa z filtrowanym sygnałem
    fig = px.line ( df , x = "index" , y = "real" , title = f"Sygnał {filename} BPSK raw: I i Q" )
    fig.add_scatter ( x = df[ "index" ] , y=df[ "imag" ] , mode = "lines" , name = "Q (imag filtrowane)" , line = dict ( dash = "dash" ) )
    fig.update_layout (
        xaxis_title = "Numer próbki" ,
        yaxis_title = "Amplituda" ,
        xaxis = dict ( rangeslider_visible = True ) ,
        legend = dict ( x = 0.01 , y = 0.99 ) ,
        height = 500
    )
    fig.show ()

def plot_symbols ( filename ) :
    # Wczytanie danych
    print("Rysuję wykres symboli BPSK...")
    df = pd.read_csv(filename)
    
    # Dodanie indeksu symboli
    df["symbol_index"] = df.index
    
    # Tworzenie wykresu
    fig = px.scatter(df, x="symbol_index", y="symbol", 
                     title=f"Symbole BPSK z pliku {filename}",
                     labels={"symbol": "Wartość symbolu", "symbol_index": "Indeks symbolu"})
    
    # Dodanie linii łączącej symbole dla lepszej wizualizacji
    fig.add_scatter(x=df["symbol_index"], y=df["symbol"], 
                    mode='lines+markers', name='Symbole BPSK',
                    line=dict(color='gray', width=1, dash='dot'))
    
    # Konfiguracja wyglądu wykresu
    fig.update_layout(
        height=500,
        xaxis=dict(rangeslider_visible=True),
        legend=dict(x=0.01, y=0.99)
    )
    
    # Ustawienie stałej skali Y dla lepszej wizualizacji BPSK
    fig.update_yaxes(range=[-1.5, 1.5])
    
    fig.show()