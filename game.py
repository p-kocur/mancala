from typing import Type
import typing 
from enum import Enum
import numpy as np

# Zmienna globalna opisująca liczbę dziur przypadających jednemu graczowi
N_HOLES = 6

# Zmienna globalna określająca ile kamyków znajduje się w jednej dziurze na początku
N_STONES = 4
          
'''
Wersja 2: dziury nie są reprezentowane jako klasy
Jest to wersja lepsza, zostajemy przy niej
'''

# Klasa reprezentująca planszę
class Board:
    def __init__(self) -> None:
        # Dziury pierwszego i drugiego gracz
        # Każda dziura ma na początku gry po N_STONES kul
        # Ponadto ostatnimi elementami w listach są tzw. "domki"
        self.first_player_holes = [N_STONES] * N_HOLES + [0]
        self.second_player_holes = [N_STONES] * N_HOLES + [0]
        
    # Metoda wizualizująca stan planszy
    def print_board(self) -> None:
        print("\n\n") 
        spare_space = "  "
        print(spare_space, end=spare_space)
        for i in range(N_HOLES-1, -1, -1):
            print(str(self.second_player_holes[i]).rjust(2), end=spare_space)    
        print()
        print(str(self.second_player_holes[-1]).rjust(2), end=spare_space)
        for i in range(N_HOLES):
            print(spare_space, end=spare_space)
        print(str(self.first_player_holes[-1]).rjust(2))
        print(spare_space, end=spare_space)
        for i in range(N_HOLES):
            print(str(self.first_player_holes[i]).rjust(2), end=spare_space) 
        print("\n\n")  
            
class Game:
    def __init__(self, discount_factor = 1.0):
        self.board = Board()
        self.player = 0
        self.states = np.zeros((1, 2*N_HOLES+2))
        self.actions = [0, 1, 2, 3, 4, 5]
        self.discount_factor = discount_factor
    
    def terminal(self):
        return self.is_finished() or self.board.first_player_holes[-1] > 24 or self.board.second_player_holes[-1] > 24
 
    def init_state(self):
        self.reset()
        return self.state()   
    
    def sim_transition(self, a):
        reward = self.reward_fn()
        if a != None:
            self.player = self.move(self.player, a+1)
        return (reward,
                self.init_state() if self.terminal() else
                    self.state())  
        
    def play_ai(self, ai_policy) -> None:
        self.reset()
        self.player = 0
        
        # Dopóki gra nie jest zakończona
        while not self.is_finished():
            self.board.print_board()
            if self.player == 0:
                print(f"Ruch gracza {self.player+1}")
                hole_n = int(input("Wybierz numer dziury: "))
            else:
                print(f"Ruch gracza AI")
                hole_n = ai_policy(self.state())+1
                
            self.player = self.move(self.player, hole_n)
            
        self.board.print_board()
        # Zwracamy informacje o wygranej
        if self.board.first_player_holes[-1] > self.board.second_player_holes[-1]:
            print("\n\nZwyciężył gracz nr 1!")
        elif self.board.first_player_holes[-1] < self.board.second_player_holes[-1]:
            print("\n\nZwyciężył AI")
        else:
            print("\n\nRemis!")
        
    # Metoda tworząca rozgrywkę w konsoli
    def play(self) -> None:
        player = 0
        
        # Dopóki gra nie jest zakończona
        while not self.terminal():
            self.board.print_board()
            print("\n\n")
            print(f"Ruch gracza {player+1}")
            hole_n = int(input("Wybierz numer dziury: "))
            player = self.move(player, hole_n)
            
        self.board.print_board()
        # Zwracamy informacje o wygranej
        if self.board.first_player_holes[-1] > self.board.second_player_holes[-1]:
            print("\n\nZwyciężył gracz nr 1!")
        elif self.board.first_player_holes[-1] < self.board.second_player_holes[-1]:
            print("\n\nZwyciężył gracz nr 2!")
        else:
            print("\n\nRemis!")
            
    # Metoda sprawdzająca, czy gra jest zakończona  
    # Jeśli gra została skończona, opróżniamy planszę z kul według zasad
    def is_finished(self) -> bool:
        if sum(self.board.first_player_holes[:N_HOLES]) == 0:
            for i in range(N_HOLES+1):
                self.board.second_player_holes[-1] += self.board.second_player_holes[i]
                self.board.second_player_holes[i] = 0
            return True
        elif sum(self.board.second_player_holes[:N_HOLES]) == 0:
            for i in range(N_HOLES+1):
                self.board.first_player_holes[-1] += self.board.first_player_holes[i]
                self.board.first_player_holes[i] = 0
            return True
        else:
            return False
    
    # Metoda odpowiadająca za wykonanie ruchu
    # Zwraca informację o tym, kto wykonuje kolejny ruch
    def move(self, player: int, hole_n: int) -> int:
        # Ustalamy, które dziury są nasze, a które przeciwnika
        our_holes = self.board.first_player_holes if player == 0 else self.board.second_player_holes
        opponent_holes = self.board.second_player_holes if player == 0 else self.board.first_player_holes
        # Kontrolujemy ostatnią dziurę, do której wrzuciliśmy kulę 
        last = False
        # Indeks wybranej dziury
        idx = hole_n - 1
        previous_h_n = our_holes[idx] 
        our_holes[idx] = 0 
        
        # Proces wykonywania ruchu
        i = idx+1
        for _ in range(previous_h_n):
            # Jeśli dotarliśmy do domku przeciwnika, indeksujemy od początku
            if i == 2*N_HOLES + 1:
                i = 0
                
            # Jeśli kulka spada po naszej stronie
            if i // (N_HOLES + 1) == 0:
                our_holes[i] += 1  
            # Jeśli kulka spada po stronie przeciwnika
            else:
                opponent_holes[i % (N_HOLES + 1)] += 1
            
            # Jeśli kulka spadła do naszego domku
            if i == N_HOLES:
                last = True
            else:
                last = False
            
            # Przechodzimy do kolejnej dziury    
            i += 1
              
        # Jeśli ostatni kamień spadł do domku, mamy kolejny ruch
        if last is True:
            return player
        # Jeśli nie, to sprawdzamy czy doszło do przejęcia kulek przeciwnika
        elif i - 1 < N_HOLES and our_holes[i-1] == 1 and opponent_holes[N_HOLES-i] != 0:
            our_holes[-1] += 1 + opponent_holes[N_HOLES-i]
            our_holes[i-1] = 0
            opponent_holes[N_HOLES-i] = 0
        return int(not player)
    
    # Zwraca stan planszy w formie listy 
    def state(self) -> np.ndarray:
        fp_holes = self.board.first_player_holes[:]
        sp_holes = self.board.second_player_holes[:]
        
        if self.player == 0:
            fp_holes.extend(sp_holes)
            to_return = np.array([[fp_holes]])
        else:
            sp_holes.extend(fp_holes)
            to_return = np.array([[sp_holes]])
            
        return to_return
    
    # Resetuje grę
    def reset(self) -> None:        
        self.board.first_player_holes = [N_STONES] * N_HOLES + [0]
        self.board.second_player_holes = [N_STONES] * N_HOLES + [0]
       
    # Funkcja określająca nagrodę 
    def reward_fn(self) -> int:
        if not self.is_finished or self.board.first_player_holes[-1] == self.board.second_player_holes[-1]:
            return 0
            
        if self.player == 0:
            if self.board.first_player_holes[-1] > self.board.second_player_holes[-1]:
                return 1
            else:
                return -1
        else:
            if self.board.first_player_holes[-1] > self.board. second_player_holes[-1]:
                return -1
            else:
                return 1
            
            
             
def main():
    game = Game()
    game.play()
       
if __name__ == "__main__":
    main()
        