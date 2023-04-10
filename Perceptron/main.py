import csv
import random
import numpy as np


class Perceptron:

    # Konstruktor
    def __init__(self, stala_uczenia=0.01):
        # Im mniejsza stała uczenia, tym mniejsza konwergencja, ale dokładniejsze wyniki!
        self.stala_uczenia = stala_uczenia
        # Wagi na none, bo będą inicjalizowane podczas procesu uczenia
        self.wagi = None

    # Wczytanie dowolnego zbioru treningowego z pliku w formacie csv, gdzie ostatnia kolumna to atrybut decyzyjny.
    # Powinien dostosowywać liczbę wag do wczytanego zbioru.
    @staticmethod
    def wczytaj_dane(sciezka_pliku):
        # Przechowywanie dataset'u z podmienionymi nazwami na liczby
        dane = []
        # Czytanie z zapewnionego pliku
        with open(sciezka_pliku, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # Dla każdego rzędu w pliku
            for row in reader:
                # Podmień ostatnią kolumnę na 0 lub 1 w zależności od tego, co się w niej znajduje
                podmiana = 0 if row[-1] == 'Iris-versicolor' else 1
                dane.append([float(x) for x in row[:-1]] + [podmiana])
        return dane

    def aktywacja(self, wartosci):
        # dot — produkt wewnętrzny a1b1 + a2b2 ...
        return 1 if np.dot(self.wagi, wartosci) >= 0 else 0

    def zaklasyfikuj(self, wiersz):
        # Z dodaniem bias
        inputs = np.array([1] + wiersz)
        return self.aktywacja(inputs)

    def trenowanie(self, dane_treningowe, liczba_epok=100):
        # Nie chcemy ostatniej kolumny, bo jest to tekst!
        dlugosc_bez_nazwy = len(dane_treningowe[0]) - 1

        # Ustawienie losowej wagi z zakresu od 0 do 1 - dla pierwszej z wag
        # Pozwala to na zachowanie względnej neutralności jednocześnie nie stojąc w lokalnym minimum
        self.wagi = [random.uniform(0, 1)] + [0.0 for _ in range(dlugosc_bez_nazwy)]
        # print(self.wagi)

        for epoka in range(liczba_epok):
            for wiersz in dane_treningowe:
                # Dodajemy na start 1 do inputu — jako bias i ucinamy ostatnią kolumnę
                inputs = np.array([1] + wiersz[:-1])
                # To, co chcielibyśmy uzyskać — czytane prosto z wiersza
                oczekiwany_output = wiersz[-1]
                # Wyliczanie aktywacji
                uzyskany_output = self.aktywacja(inputs)
                # Wyliczanie błędu
                blad = oczekiwany_output - uzyskany_output
                # Korekcja błędu
                self.wagi += self.stala_uczenia * blad * inputs


def main():
    stala = float(input("Wprowadź stałą uczenia (dla najlepszych efektów wybierz małą stałą - rzędu 0.01): "))
    liczba_epok = int(input("Wprowadź liczbę epok: "))

    perceptron = Perceptron(stala_uczenia=stala) # 0.01

    dane_treningowe = perceptron.wczytaj_dane("perceptron.data")
    dane_testowe = perceptron.wczytaj_dane("perceptron.test.data")

    perceptron.trenowanie(dane_treningowe, liczba_epok=liczba_epok) # 100

    # dodawanie 1 - jeżeli doszło do prawidłowej klasyfikacji
    liczba_prawidlowych_przypisan = sum(1 for row in dane_testowe if perceptron.zaklasyfikuj(row[:-1]) == row[-1])

    dokladnosc = liczba_prawidlowych_przypisan / len(dane_testowe)
    print(f'Dokładność klasyfikacji: {dokladnosc}')

    # Mapping by pokazywało nazwę jak mapujemy
    mapping = {0: "Iris-versicolor", 1: "Iris-virginica"}

    # INPUT UŻYTKOWNIKA
    while True:
        input_uzytkownika = input("Wprowadź dane oddzielone przecinkami, lub wpisz 'q' by wyjść: ")
        if input_uzytkownika.lower() == "q":
            break
        # Dzielenie po przecinkach
        wektor = [float(x) for x in input_uzytkownika.split(',')]
        # Klasyfikacja
        rezultat = perceptron.zaklasyfikuj(wektor)
        print(f'Zaklasyfikowano jako: {mapping[rezultat]}')


if __name__ == '__main__':
    main()
