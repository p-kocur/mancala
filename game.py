from typing import Type
import typing
from enum import Enum

# Zmienna globalna opisująca liczbę dziur przypadających jednemu graczowi
N_HOLES = 6

# Zmienna globalna określająca ile kamyków znajduje się w jednej dziurze na początku
N_STONES = 4

# Klasa typu Enum, w której znajdują się dwa możliwe typy graczy
class Player(Enum):
    FIRST = 0
    SECOND = 1

# Klasa reprezentująca "dziurę" na planszy
class Hole():
    
    # Zmienna klasowa
    n_holes = 0
    
    def __init__(self, owner: Player=None, n_stones: int=N_STONES) -> None:
        Hole.n_holes += 1
        
        self.ID = Hole.n_holes
        self.owner = owner
        self.n_stones = n_stones
        self.next = None
        self.opposite = None
        
    def is_empty(self) -> bool:
        return self.n_stones == 0
        
# Klasa reprezentująca planszę
class Board:
    def __init__(self) -> None:
        
        # Domek gracza pierwszego
        self.home_first = Hole(Player.FIRST, 0)
        
        # Domek gracza drugiego
        self.home_second = Hole(Player.SECOND, 0)
        
        # Dziury gracza pierwszego
        self.holes_first = [Hole(Player.FIRST)]
        
        # Dziury gracza drugiego
        self.holes_second = [Hole(Player.SECOND)]
        
        # Dodajemy kolejne N_HOLES - 1 dziur dla każdego gracza
        for i in range(N_HOLES-1):
            self.holes_first.append(Hole(Player.FIRST))
            self.holes_first[-2].next = self.holes_first[-1]
            
            self.holes_second.append(Hole(Player.SECOND))
            self.holes_second[-2].next = self.holes_second[-1]
            
        # Łączymy domki z resztą dziur
        self.home_first.next = self.holes_second[0]
        self.home_second.next = self.holes_first[0]
        
        self.holes_first[-1].next = self.home_first
        self.holes_second[-1].next = self.home_second
        
        # Ustalamy, które dziury są naprzeciwko siebie
        for i in range(N_HOLES):
            self.holes_first[i].opposite = self.holes_second[-(i+1)]
            self.holes_second[i].opposite = self.holes_first[-(i+1)]
        
    def move(self, player: Player, i: int) -> int:
        if player == Player.FIRST:
            holes = self.holes_first
            home = self.home_first
        else:
            holes = self.holes_second
            home = self.home_second
            
        if holes[i].is_empty():
            return 2
        
        if i < N_HOLES - 1:
            holes
            
        
            
        
            
        
            
            
            
        
        
def main():
    pass
       
if __name__ == "__main__":
    main()
        