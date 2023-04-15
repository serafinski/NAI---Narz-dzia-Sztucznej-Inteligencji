import csv
import numpy as np


class Perceptron:
    def __init__(self, stala_uczenia=0.01):
        self.stala_uczenia = stala_uczenia
        self.wagi = None

    @staticmethod
    def wczytaj_dane(sciezka_pliku):
        dane = []
        jezyki_mapping = {"English": 0, "German": 1, "Polish": 2, "Spanish": 3}  # Add a mapping dictionary
        with open(sciezka_pliku, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for rzad in reader:
                jezyk, tekst = rzad[0], rzad[1]  # Change this line to reflect the correct column indices
                tekst = tekst.strip('"')  # Remove the surrounding double quotes
                jezyk = jezyki_mapping[jezyk.strip().strip('"')]  # Map the jezyk string to its corresponding integer
                dane.append((tekst, jezyk))
        return dane

    def pre_processing(self, tekst):
        tekst = tekst.lower()
        wektor_znakow = np.zeros(36)  # Increased vector size to accommodate digits
        for znak in tekst:
            if 'a' <= znak <= 'z':
                wektor_znakow[ord(znak) - ord('a')] += 1
            elif '0' <= znak <= '9':  # Added condition to include digits
                wektor_znakow[26 + ord(znak) - ord('0')] += 1
        wektor_normalizacja = wektor_znakow / np.linalg.norm(wektor_znakow)
        return wektor_normalizacja

    def aktywacja(self, wartosci):
        z = np.dot(self.wagi, wartosci)
        return 1 / (1 + np.exp(-z))

    def zaklasyfikuj(self, wiersz):
        inputs = self.pre_processing(wiersz)
        aktywacje = self.aktywacja(inputs)
        return np.argmax(aktywacje)

    def trenowanie(self, dane_treningowe, liczba_epok=100, liczba_jezykow=4):
        self.wagi = np.zeros((liczba_jezykow, 36))  # Update the dimensions to (num_languages, 36)
        epsilon = 1e-8  # Add a small constant to prevent division by zero
        for epoka in range(liczba_epok):
            for wiersz, oczekiwany_output in dane_treningowe:
                inputs = self.pre_processing(wiersz)
                uzyskany_output = self.aktywacja(inputs)
                blad = np.zeros(liczba_jezykow)
                blad[oczekiwany_output] = 1
                blad -= uzyskany_output
                blad *= uzyskany_output * (1 - uzyskany_output)  # Add this line to calculate the gradient
                self.wagi += self.stala_uczenia * np.outer(blad, inputs)
                self.wagi = self.wagi / (np.linalg.norm(self.wagi, axis=1, keepdims=True) + epsilon)  # Add epsilon to the norm


def main():
    stala_uczenia = float(input("Wprowadź stałą uczenia (dla najlepszych efektów wybierz małą stałą - rzędu 0.01): "))
    liczba_epok = int(input("Wprowadź liczbę epok: "))

    perceptron = Perceptron(stala_uczenia=stala_uczenia)

    dane_treningowe = perceptron.wczytaj_dane("lang.train.csv")
    dane_testowe = perceptron.wczytaj_dane("lang.test.csv")

    perceptron.trenowanie(dane_treningowe, liczba_epok=liczba_epok)

    prawidlowe_przypisania = sum(1 for rzad, jezyk in dane_testowe if perceptron.zaklasyfikuj(rzad) == jezyk)

    dokladnosc = prawidlowe_przypisania / len(dane_testowe)
    print(f'Dokładność klasyfikacji: {dokladnosc}')

    mapping = {0: "Angielski", 1: "Niemiecki", 2: "Polski", 3: "Hiszpański"}

    while True:
        input_uzytkownika = input("Wpisz tekst do zaklasyfikowania, lub wpisz 'q' by wyjść: ")
        if input_uzytkownika.lower() == "q":
            break
        rezultat = perceptron.zaklasyfikuj(input_uzytkownika)
        print(f'Zaklasyfikowany jako język: {mapping[rezultat]}')


if __name__ == '__main__':
    main()
