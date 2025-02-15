import numpy as np 
import random 
from time import sleep


#Define los estados SS como todas las configuraciones válidas del tablero.

class TicTacToe: 

    def crear_tablero(self, tamanio):
        tablero = np.zeros((tamanio, tamanio), dtype=int)  
        return tablero

    # Check for empty places on tablero
    def posibilidades(self, tablero):
        l = []
        for i in range(len(tablero)):
            for j in range(len(tablero)):
                if tablero[i][j] == 0:
                    l.append((i, j))
        return l

# Select a random place for the jugador
    def random_place(self, tablero, jugador):
        selection = self.posibilidades(tablero)
        posicion = random.choice(selection)
        tablero[posicion] = jugador
        return(tablero)

    # Checks whether the jugador has three
    # of their marks in a horizontal row 
    def row_win(self, tablero, jugador):
        for x in range(len(self.tablero)):
            win = True
            for y in range(len(self.tablero)):
                if self.tablero[x][y] != self.jugador:
                    win = False
                    continue
    
            if win == True:
                return(win)
        return(win)

# Checks whether the jugador has three
    # of their marks in a horizontal row 
    def col_win(self, tablero, jugador):
        for x in range(len(self.tablero)):
            win = True
            for y in range(len(self.tablero)):
                if self.tablero[y][x] != self.jugador:
                    win = False
                    continue
    
            if win == True:
                return(win)
        return(win)

    # Checks whether the jugador has three
    # of their marks in a diagonal row
    def diag_win(self, tablero, jugador):
        # Verificar la diagonal principal
        win = True
        for x in range(len(tablero)):
            if tablero[x, x] != jugador:
                win = False
                break
        if win:
            return True
        
        # Verificar la diagonal secundaria
        win = True
        for x in range(len(tablero)):
            y = len(tablero) - 1 - x
            if tablero[x][y] != jugador:
                win = False
                break
        return win

    # Evaluates whether there is
    # a winner or a tie
    
    
    def evaluate(self, tablero):
        winner = 0

        for self.jugador in [1, 2]:
            if (self.row_win(self.tablero, self.jugador) or
                    self.col_win(self.tablero, self.jugador) or
                    self.diag_win(self.tablero, self.jugador)):

                winner = self.jugador

        if np.all(tablero != 0) and winner == 0:
            winner = -1
        return winner

    def play_game(self):
        self.tablero, winner, counter = self.crear_tablero(3), 0, 1
        print("Tablero inicial")
        print(self.tablero)
        sleep(2)
    
        while winner == 0:
            for self.jugador in [1, 2]:
                self.tablero = self.random_place(self.tablero, self.jugador)
                print("\n Tablero después de " + str(counter) + " movimiento")
                print(self.tablero)
                sleep(2)
                counter += 1
                winner = self.evaluate(self.tablero)
                if winner != 0:
                    break
        return(winner)

game = TicTacToe()
tablero = game.crear_tablero(3)

# print(game.posibilidades(tablero))
# print(game.random_place(tablero, 1))

print("El ganador es: " + str(game.play_game()))
