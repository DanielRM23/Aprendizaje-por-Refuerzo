import numpy as np
import random

class TicTacToe:
    def __init__(self, tamanio=3):
        self.tamanio = tamanio #Tamaño del tablero 
        self.tablero = self.crear_tablero() #Se crea el tablero 
        self.estados_valores = {}  # Diccionario para almacenar los valores de los estados
        self.politica = {}  # Diccionario para almacenar la política óptima
    
    def crear_tablero(self):
        # Se crea un tablero con dimensiones taminio x tamanio
        # El tablero es una matriz de ceros de dicho tamaño, donde cada cero representa una posición vacía en el tablero
        return np.zeros((self.tamanio, self.tamanio), dtype=int) 

    def obtener_estado(self, tablero):
        # Esto se hace con el fin utilizar el estado del tablero como clave en el diccionario de valores de estado
        return tuple(map(tuple, tablero))

    def posibilidades(self, tablero):
        # Devuelve una lista de todas las posiciones vacías en el tablero
        # Esto se traduce a verificar que en la posición (i,j) del tablero haya un 0 (cero)
        posiciones = []
        for i in range(self.tamanio): 
            for j in range(self.tamanio):
                if tablero[i][j] == 0:
                    posiciones.append((i,j))

        return posiciones

    def aplicar_accion(self, tablero, accion, jugador):
        # Crea una copia del tablero original para no modificar el tablero existente directamente.
        # Luego, actualiza la posición especificada por la acción con el número del jugador.
        # Finalmente, devuelve el nuevo tablero con la acción aplicada.
        nuevo_tablero = tablero.copy()
        nuevo_tablero[accion] = jugador
        return nuevo_tablero


    def row_win(self, tablero, jugador):
        for x in range(len(tablero)):  # Recorremos cada fila
            win = True  # Inicializamos la variable win como True
            for y in range(len(tablero[0])):  # Recorremos cada columna en la fila
                if tablero[x][y] != jugador:  # Si encontramos un valor diferente al del jugador
                    win = False  # Cambiamos win a False
                    break  # Salimos del ciclo
            if win:  # Si no se encontró un valor diferente, significa que el jugador ha ganado en esta fila
                return True
        return False  # Si no hay victoria en ninguna fila, devolvemos False
    
    def col_win(self, tablero, jugador):
        for x in range(len(tablero[0])):  # Recorremos cada columna
            win = True  # Inicializamos la variable win como True
            for y in range(len(tablero)):  # Recorremos cada fila en la columna
                if tablero[y][x] != jugador:  # Si encontramos un valor diferente al del jugador
                    win = False  # Cambiamos win a False
                    break  # Salimos del ciclo
            if win:  # Si no se encontró un valor diferente, significa que el jugador ha ganado en esta columna
                return True
        return False  # Si no hay victoria en ninguna columna, devolvemos False

    def diag_win(self, tablero, jugador):
        # Verificar la diagonal principal
        win = True
        for x in range(len(tablero)):
            if tablero[x][x] != jugador:  # Si encontramos un valor diferente en la diagonal principal
                win = False  # Cambiamos win a False
                break  # Salimos del ciclo
        if win:  # Si no se encontró un valor diferente, significa que el jugador ha ganado en la diagonal principal
            return True

        # Verificar la diagonal secundaria
        win = True
        for x in range(len(tablero)):
            y = len(tablero) - 1 - x  # Calculamos el índice de la columna de la diagonal secundaria
            if tablero[x][y] != jugador:  # Si encontramos un valor diferente en la diagonal secundaria
                win = False  # Cambiamos win a False
                break  # Salimos del ciclo
        return win  # Retornamos si el jugador ha ganado en la diagonal secundaria

    def es_terminal(self, tablero):
        for jugador in [1, 2]:
            if self.row_win(tablero, jugador) or self.col_win(tablero, jugador) or self.diag_win(tablero, jugador):
                return jugador

        if not self.posibilidades(tablero):
            return -1  # No hay más movimientos, el juego terminó en empate
        else:
            return 0  # El juego puede continuar, hay movimientos posibles