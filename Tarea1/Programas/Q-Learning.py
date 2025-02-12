#!/usr/bin/env python
# coding: utf-8

# Q-Learning
# Daniel Rojo Mata



import numpy as np
import random

# Clase que define el entorno de juego Lago Congelado
class LagoCongelado:
    def __init__(self, tamano=4):
        # Inicialización del tablero con dimensiones tamano x tamano
        self.tamano = tamano  
        self.matriz = np.zeros((tamano, tamano), dtype=int)  # matriz de ceros representa el tablero vacío
        self.estado_inicial = (0, 0)  # posición inicial del agente
        self.estado_meta = (tamano-1, tamano-1)  # posición de la meta (esquina inferior derecha)
        self.estados_hoyo = [(1, 1), (1, 3), (2, 3), (3, 0)]  # lista de posiciones peligrosas (hoyos)
        self.inicializar_hoyos()  # coloca los hoyos en el tablero


    # Coloca los hoyos en las posiciones indicadas en la lista estados_hoyo
    def inicializar_hoyos(self):
        for i, j in self.estados_hoyo:
            self.matriz[i][j] = 1  # marca las celdas peligrosas con un 1 en la matriz


    # Reinicia el entorno para comenzar un nuevo episodio
    def reiniciar(self):
        self.estado_actual = self.estado_inicial  # el agente vuelve a la posición inicial
        return self.estado_actual


    # Realiza un paso de acción y determina el nuevo estado, recompensa y si el episodio terminó
    def paso(self, accion):
        i, j = self.estado_actual  # obtiene la posición actual del agente
        i, j = self.actualizar_posicion(i, j, accion)  # actualiza la posición en función de la acción
        self.estado_actual = (i, j)  # actualiza el estado actual del agente
        recompensa, terminado = self.determinar_resultado()  # determina recompensa y si el episodio ha terminado
        return self.estado_actual, recompensa, terminado  # retorna el nuevo estado, recompensa y si terminó


    # Actualiza la posición del agente basado en la acción seleccionada
    def actualizar_posicion(self, i, j, accion):
        if accion == 0:  # acción de moverse arriba
            i = max(i-1, 0)
        elif accion == 1:  # acción de moverse abajo
            i = min(i+1, self.tamano-1)
        elif accion == 2:  # acción de moverse a la izquierda
            j = max(j-1, 0)
        elif accion == 3:  # acción de moverse a la derecha
            j = min(j+1, self.tamano-1)
        return i, j  # retorna la nueva posición (i, j)
    

    # Determina la recompensa y si el episodio ha terminado según el estado actual
    def determinar_resultado(self):
        if self.estado_actual == self.estado_meta:  # si el agente alcanzó la meta
            return 1, True  # recompensa positiva y el episodio termina
        elif self.estado_actual in self.estados_hoyo:  # si el agente cae en un hoyo
            return -1, True  # recompensa negativa y el episodio termina
        else:
            return 0, False  # si no hay evento especial, no hay recompensa y el episodio sigue
    

    # Muestra el estado actual del tablero
    def mostrar(self):
        print('\n')
        for i in range(self.tamano):
            for j in range(self.tamano):
                self.mostrar_celda(i, j)  # muestra el contenido de cada celda
            print()
        print()
    

    # Muestra el contenido de cada celda en función del estado actual del agente, meta y hoyos
    def mostrar_celda(self, i, j):
        if self.matriz[i][j] == 0:
            if (i, j) == self.estado_actual:
                print('A', end=' ')  # muestra la posición del agente
            elif (i, j) == self.estado_meta:
                print('M', end=' ')  # muestra la posición de la meta
            else:
                print('.', end=' ')  # muestra camino seguro
        elif self.matriz[i][j] == 1:
            print('A' if (i, j) == self.estado_actual else 'X', end=' ')  # muestra hoyo o agente en un hoyo


    # Muestra la tabla Q para cada estado posible
    def mostrar_q_tabla(self, tabla_q):
        print('-----------------------------------------------------------------')
        print('Tabla-Q:')
        print('-----------------------------------------------------------------')
        for i in range(self.tamano):
            for j in range(self.tamano):
                self.mostrar_valores_q(i, j, tabla_q)  # muestra valores Q para cada acción en el estado


    # Muestra los valores Q para un estado o indica "NULO" si es un hoyo
    def mostrar_valores_q(self, i, j, tabla_q):
        if self.matriz[i][j] == 0:
            for accion in range(4):
                print('%.2f' % tabla_q[i][j][accion], end='\t')  # imprime el valor Q para cada acción
            print()
        else:
            print('NULO', 'NULO', 'NULO', 'NULO', sep='\t')  # muestra NULO para cada acción en un hoyo


# Define la política e-greedy para seleccionar la acción
def politica_egreedy(estado, epsilon):
    if random.uniform(0, 1) < epsilon:  # selecciona acción aleatoria para explorar
        return random.randint(0, 3)
    return np.argmax(tabla_q[estado[0]][estado[1]])  # selecciona la mejor acción según la tabla Q


# Reduce el valor de epsilon después de cada episodio para reducir la exploración
def actualizar_epsilon(epsilon, min_epsilon, tasa_reduccion_epsilon):
    return max(min_epsilon, epsilon * (1 - tasa_reduccion_epsilon))


# Entrena al agente usando el algoritmo Q-Learning
def entrenar_agente(ambiente, num_episodios, max_pasos_por_episodio, tasa_aprendizaje, factor_descuento, epsilon, min_epsilon, tasa_reduccion_epsilon):
    for episodio in range(num_episodios):
        estado = ambiente.reiniciar()  # reinicia el entorno al inicio de cada episodio
        terminado = False
        pasos = 0
        while not terminado and pasos < max_pasos_por_episodio:
            accion = politica_egreedy(estado, epsilon)  # selecciona una acción basada en e-greedy
            nuevo_estado, recompensa, terminado = ambiente.paso(accion)  # realiza la acción y recibe los resultados
            # Actualiza la tabla Q utilizando la ecuación de Q-Learning
            tabla_q[estado[0]][estado[1]][accion] += tasa_aprendizaje * \
                (recompensa + factor_descuento * np.max(tabla_q[nuevo_estado[0]][nuevo_estado[1]]) - tabla_q[estado[0]][estado[1]][accion])
            estado = nuevo_estado  # actualiza el estado
            pasos += 1
        epsilon = actualizar_epsilon(epsilon, min_epsilon, tasa_reduccion_epsilon)  # reduce epsilon

        # Muestra el progreso cada 1000 episodios
        if episodio % 1000 == 0:
            print(f"Episodio {episodio}")
            ambiente.mostrar()
            ambiente.mostrar_q_tabla(tabla_q)


# Ejecuta la política aprendida después del entrenamiento
def ejecutar_politica_aprendida(ambiente, tabla_q):
    estado = ambiente.reiniciar() 
    terminado = False
    pasos = 0
    print("\nEl agente está corriendo lo aprendido..\n")
    ambiente.mostrar()

    while not terminado:
        accion = np.argmax(tabla_q[estado[0]][estado[1]])  # selecciona la mejor acción de la tabla Q
        nuevo_estado, recompensa, terminado = ambiente.paso(accion)  # realiza la acción y recibe los resultados
        ambiente.mostrar()  # muestra el entorno después de cada paso
        print(f"Paso {pasos + 1}: Estado {estado} -> Acción {['ARRIBA', 'ABAJO', 'IZQUIERDA', 'DERECHA'][accion]}")
        estado = nuevo_estado
        pasos += 1
        if terminado:  # muestra un mensaje final según el resultado (meta o hoyo)
            print("\n¡El agente llegó a la meta!" if recompensa == 1 else "\n¡El agente se cayó en un hoyo!")
            break

def calcular_valor_V(tabla_q):
    V = np.zeros((ambiente.tamano, ambiente.tamano))
    for i in range(ambiente.tamano):
        for j in range(ambiente.tamano):
            V[i][j] = np.max(tabla_q[i][j])  # Tomamos el máximo valor Q para cada estado
    return V

def mostrar_valor_V(V):
    print('\n-----------------------------------------------------------------')
    print('Valor V(s):')
    print('-----------------------------------------------------------------')
    for i in range(ambiente.tamano):
        for j in range(ambiente.tamano):
            print('%.2f' % V[i][j], end='\t')
        print()

def obtener_politica_optima(tabla_q):
    politica_optima = np.zeros((ambiente.tamano, ambiente.tamano), dtype=int)
    for i in range(ambiente.tamano):
        for j in range(ambiente.tamano):
            # Selecciona la acción con el mayor valor Q para el estado (i, j)
            politica_optima[i][j] = np.argmax(tabla_q[i][j])
    return politica_optima


def mostrar_politica_optima(politica_optima):
    print('-----------------------------------------------------------------')
    print('Política Óptima:')
    print('-----------------------------------------------------------------')
    acciones = ['↑', '↓', '←', '→']  # Representación de las acciones
    for i in range(ambiente.tamano):
        for j in range(ambiente.tamano):
            if ambiente.matriz[i][j] == 1:  # Si es un hoyo
                print('X', end='\t')
            elif (i, j) == ambiente.estado_meta:  # Si es la meta
                print('M', end='\t')
            else:
                print(acciones[politica_optima[i][j]], end='\t')  # Muestra la acción óptima
        print()

######################## Ejecución del algoritmo ########################


# Inicialización del entorno de juego
ambiente = LagoCongelado()  # crea una instancia del entorno Lago Congelado con el tamaño por defecto

# Inicialización de la tabla Q, que almacena los valores de cada acción para cada estado
tabla_q = np.zeros((ambiente.tamano, ambiente.tamano, 4))  # tabla Q con valores iniciales de 0, con dimensiones (tamaño del tablero, tamaño del tablero, número de acciones)

# Hiperparámetros para el entrenamiento
num_episodios = 10000  # número total de episodios de entrenamiento
max_pasos_por_episodio = 100  # máximo número de pasos permitidos por episodio
tasa_aprendizaje = 0.1  # tasa de aprendizaje que controla el ajuste de los valores Q
factor_descuento = 0.99  # factor de descuento que prioriza las recompensas futuras sobre las inmediatas
epsilon = 1.0  # tasa inicial de exploración, donde el agente explora acciones aleatorias
min_epsilon = 0.01  # valor mínimo de epsilon, asegura que siempre haya algo de exploración
tasa_reduccion_epsilon = 0.001  # tasa de reducción de epsilon, controlando la transición de exploración a explotación

# Entrenamiento del agente
entrenar_agente(ambiente, num_episodios, max_pasos_por_episodio, tasa_aprendizaje, factor_descuento, epsilon, min_epsilon, tasa_reduccion_epsilon)

# Ejecución de la política aprendida
ejecutar_politica_aprendida(ambiente, tabla_q)

# Ejecución de v(s)
V = calcular_valor_V(tabla_q)
# Mostrar v(s)
mostrar_valor_V(V)

# Obtener la política óptima
politica_optima = obtener_politica_optima(tabla_q)
# Mostrar la política óptima
mostrar_politica_optima(politica_optima)

