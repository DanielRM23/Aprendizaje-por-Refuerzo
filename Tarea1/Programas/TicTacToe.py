import numpy as np 
import random 


#Define los estados SS como todas las configuraciones válidas del tablero.

class TicTacToe: 
    # def __init__(self, tamanio = 3):
    #     # Inicialización del tablero con dimensiones: tamanio x tamanio 
    #     self.tamanio = tamanio # dimension
    #     self.matriz = np.zeros((tamano, tamano), dtype=int)  # matriz de ceros representa el tablero vacío


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


game = TicTacToe()
tablero = game.crear_tablero(3)

print(game.posibilidades(tablero))
print(game.random_place(tablero, 1))
