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
        print(tablero)



game = TicTacToe()

game.crear_tablero(3)

