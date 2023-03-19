import math
import operator

import pandas as pd
import matplotlib.pyplot as plt


def wczytaj_dane(nazwa_pliku):
    # Zwrot danych z pliku z użyciem Pandas
    dataframe = pd.read_csv(nazwa_pliku)
    # zwrot wartości dataframe w postaci NumPy array
    values = dataframe.values
    return values


def wylicz_odleglosc_euklidesowa(punkt1, punkt2, ilosc_elementow_wektora):
    odleglosc = 0
    for i in range(ilosc_elementow_wektora):
        # nawiasy do kwadratu
        odleglosc += math.pow((punkt1[i] - punkt2[i]), 2)
    # pierwiastek
    pierwiastek = math.sqrt(odleglosc)
    return pierwiastek


def zwroc_liste_sasiadow(dane_treningowe, dane_testowe, k_ilosc_sasiadow):
    # tablica tupli, w której będą przechowywane instancje testowe wraz z wyliczoną odległością
    tupla_odleglosci = []
    # bo nie chcemy nazwy
    ilosc_elementow_wektora = len(dane_testowe) - 1

    # liczenie odległości między danymi treningowymi a danymi testowymi
    for i in range(len(dane_treningowe)):
        odleglosc = wylicz_odleglosc_euklidesowa(dane_testowe, dane_treningowe[i], ilosc_elementow_wektora)
        # dodawanie do tupi
        tupla_odleglosci.append((dane_treningowe[i], odleglosc))

    # sortowanie rosnące po odległościach (elemencie na 1 indeksie w tupli)
    tupla_odleglosci.sort(key=operator.itemgetter(1))

    # zwracanie listy sąsiadów o długości zależnej od k
    sasiedzi = []

    for i in range(k_ilosc_sasiadow):
        # dodajemy tylko informacje na temat przypadku — bez odległości
        sasiedzi.append(tupla_odleglosci[i][0])

    return sasiedzi


def wylicz_najczestsza_kategorie(sasiedzi):
    # słownik, który będzie przechowywał najczęstsze kategorie z ilościami wystąpień
    slownik_kategorie = {}

    for i in range(len(sasiedzi)):
        # -1 bo ostatnim elementem jest nazwa kategorii
        kategoria = sasiedzi[i][-1]
        if kategoria in slownik_kategorie:
            slownik_kategorie[kategoria] += 1
        else:
            slownik_kategorie[kategoria] = 1

    # sortujemy od najczęstszego wystąpienia (reverse=True), itemgetter na 1, bo obchodzi nas ilość...
    posortowany_slownik = sorted(slownik_kategorie.items(), key=operator.itemgetter(1), reverse=True)

    # ... ale zwracamy samą nazwę kategorii
    najczestsza_nazwa_kategorii = posortowany_slownik[0][0]

    return najczestsza_nazwa_kategorii


def wylicz_dokladnosc(dane_testowe, prognoza):
    ilosc_poprawnych = 0
    for i in range(len(dane_testowe)):
        if dane_testowe[i][-1] == prognoza[i]:
            ilosc_poprawnych += 1

    procent = float(ilosc_poprawnych / len(dane_testowe)) * 100

    return procent


def predykcja_kategorii(dane_treningowe, dane_testowe, k):
    lista_predykcji_kategorii = []
    for i in range(len(dane_testowe)):
        # zwróć listę sąsiadów
        sasiedzi = zwroc_liste_sasiadow(dane_treningowe, dane_testowe[i], k)
        # wybierz najczęstszą kategorię z listy sąsiadów
        najczestsza_kategoria = wylicz_najczestsza_kategorie(sasiedzi)
        # dodaj do listy predykcji
        lista_predykcji_kategorii.append(najczestsza_kategoria)

    # zwróć listę z predykcjami
    return lista_predykcji_kategorii


def main(plik_dane_treningowe, plik_dane_testowe, k):
    dane_treningowe = wczytaj_dane(plik_dane_treningowe)
    dane_testowe = wczytaj_dane(plik_dane_testowe)

    predykcja = predykcja_kategorii(dane_treningowe, dane_testowe, k)

    dokladnosc = wylicz_dokladnosc(dane_testowe, predykcja)
    print("Dokładnosć wynosi: " + str(dokladnosc) + " %!")


if __name__ == '__main__':
    main('iris.data', 'iris.test.data', 3)
    main('wdbc.data', 'wdbc.test.data', 3)
