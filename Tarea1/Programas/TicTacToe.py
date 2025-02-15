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

game = TicTacToe()
tablero = game.crear_tablero(3)

print(game.posibilidades(tablero))
print(game.random_place(tablero, 1))
