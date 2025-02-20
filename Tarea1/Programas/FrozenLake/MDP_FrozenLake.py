import numpy as np

class FrozenLake:
    def __init__(self, tamano=4):
        self.tamano = tamano  # Tamaño del tablero
        self.matriz = np.zeros((tamano, tamano), dtype=int)  # Matriz que representa el tablero
        self.estado_inicial = (0, 0)  # Estado inicial
        self.estado_meta = (tamano-1, tamano-1)  # Estado meta
        self.estados_hoyo = [(1, 1), (1, 3), (2, 3), (3, 0)]  # Estados de hoyos
        self.inicializar_hoyos()  # Colocar los hoyos en el tablero

    # Coloca los hoyos en las posiciones indicadas
    def inicializar_hoyos(self):
        for i, j in self.estados_hoyo:
            self.matriz[i][j] = 1  # Marcamos los hoyos con un 1

    # Actualiza la posición del agente basado en la acción seleccionada
    def actualizar_posicion(self, i, j, accion):
        if accion == 0:  # Arriba
            i = max(i-1, 0)
        elif accion == 1:  # Abajo
            i = min(i+1, self.tamano-1)
        elif accion == 2:  # Izquierda
            j = max(j-1, 0)
        elif accion == 3:  # Derecha
            j = min(j+1, self.tamano-1)
        return i, j

    # Determina la recompensa y si el episodio ha terminado
    def determinar_resultado(self, estado):
        if estado == self.estado_meta:  # Si llegó a la meta
            return 1, True
        elif estado in self.estados_hoyo:  # Si cayó en un hoyo
            return -1, True
        else:
            return 0, False  # Si no ha terminado

    # Muestra el valor V(s)
    def mostrar_valor_V(self, V):
        print('\n-----------------------------------------------------------------')
        print('Valor V(s):')
        print('-----------------------------------------------------------------')
        for i in range(self.tamano):
            for j in range(self.tamano):
                print('%.2f' % V[i][j], end='\t')
            print()

    # Muestra la política óptima
    def mostrar_politica_optima(self, politica):
        print('-----------------------------------------------------------------')
        print('Política Óptima:')
        print('-----------------------------------------------------------------')
        acciones = ['↑', '↓', '←', '→']  # Representación de las acciones
        for i in range(self.tamano):
            for j in range(self.tamano):
                if self.matriz[i][j] == 1:  # Si es un hoyo
                    print('X', end='\t')
                elif (i, j) == self.estado_meta:  # Si es la meta
                    print('M', end='\t')
                else:
                    print(acciones[politica[i][j]], end='\t')  # Muestra la acción óptima
            print()