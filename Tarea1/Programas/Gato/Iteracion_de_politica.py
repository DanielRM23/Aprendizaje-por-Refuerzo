import numpy as np 
import random  

    # --------------------------- Evaluación de política ---------------------------

class TicTacToeAI:
    def __init__(self, juego):
        """
        Inicializa la IA con una instancia del juego TicTacToe.
        - juego: Instancia de la clase TicTacToe.
        - estados_valores: Diccionario para almacenar los valores de los estados.
        - politica: Diccionario para almacenar la política óptima.
        """
        self.juego = juego  # Guarda la instancia del juego.
        self.estados_valores = {}  # Diccionario para almacenar los valores de los estados.
        self.politica = {}  # Diccionario para almacenar la política óptima.

    def inicializar_politica(self):
        """
        Inicializa la política con acciones aleatorias para cada estado.
        Recorre todos los estados en `estados_valores` y asigna una acción aleatoria
        si hay movimientos posibles en ese estado.
        """
        for estado in self.estados_valores:  # Recorre cada estado en el diccionario de valores.
            acciones_posibles = self.juego.posibilidades(np.array(estado))  # Obtiene las acciones posibles para el estado.
            if acciones_posibles:  # Si hay acciones posibles:
                self.politica[estado] = random.choice(acciones_posibles)  # Asigna una acción aleatoria a la política.

    def inicializar_valores_estado(self):
        """
        Inicializa los valores de todos los estados a cero.
        """
        for estado in self.estados_valores:  # Recorre cada estado en el diccionario de valores.
            self.estados_valores[estado] = 0  # Inicializa el valor del estado a cero.

    def evaluar_politica(self, gamma=0.9, theta=1e-6):
        """
        Evalúa la política actual calculando los valores de los estados.
        - gamma: Factor de descuento (por defecto 0.9).
        - theta: Umbral de convergencia (por defecto 1e-6).
        """
        while True:  # Bucle infinito hasta que se alcance la convergencia.
            delta = 0  # Inicializa la diferencia máxima entre iteraciones.
            for estado in self.estados_valores:  # Recorre cada estado en el diccionario de valores.
                v = self.estados_valores[estado]  # Guarda el valor actual del estado.
                accion = self.politica[estado]  # Obtiene la acción según la política actual.
                nuevo_tablero = self.juego.aplicar_accion(np.array(estado), accion, 1)  # Aplica la acción al estado actual.
                nuevo_estado = self.juego.obtener_estado(nuevo_tablero)  # Obtiene el nuevo estado después de la acción.
                recompensa = self.juego.es_terminal(nuevo_tablero)  # Calcula la recompensa del nuevo estado.

                # Define la recompensa según el resultado del juego:
                if recompensa == 1:
                    recompensa = 1  # El agente gana.
                elif recompensa == 2:
                    recompensa = -1  # El oponente gana.
                elif recompensa == -1:
                    recompensa = 0  # Empate.
                else:
                    recompensa = 0  # Juego en progreso.

                # Actualiza el valor del estado usando la ecuación de Bellman:
                self.estados_valores[estado] = recompensa + gamma * self.estados_valores.get(nuevo_estado, 0)
                delta = max(delta, abs(v - self.estados_valores[estado]))  # Actualiza la diferencia máxima.

            # Si la diferencia entre iteraciones es menor que theta, se detiene:
            if delta < theta:
                break

    def mejorar_politica(self, gamma=0.9):
        """
        Mejora la política actual seleccionando la acción que maximiza el valor esperado.
        - gamma: Factor de descuento (por defecto 0.9).
        Retorna True si la política es estable, False si cambió.
        """
        politica_estable = True  # Inicializa la variable que indica si la política es estable.
        for estado in self.estados_valores:  # Recorre cada estado en el diccionario de valores.
            acciones_posibles = self.juego.posibilidades(np.array(estado))  # Obtiene las acciones posibles para el estado.
            if acciones_posibles:  # Si hay acciones posibles:
                mejor_accion = None  # Inicializa la mejor acción.
                mejor_valor = -float('inf')  # Inicializa el mejor valor con un número muy pequeño.

                # Busca la mejor acción para el estado actual:
                for accion in acciones_posibles:  # Recorre cada acción posible.
                    nuevo_tablero = self.juego.aplicar_accion(np.array(estado), accion, 1)  # Aplica la acción.
                    nuevo_estado = self.juego.obtener_estado(nuevo_tablero)  # Obtiene el nuevo estado.
                    recompensa = self.juego.es_terminal(nuevo_tablero)  # Calcula la recompensa.

                    # Define la recompensa según el resultado del juego:
                    if recompensa == 1:
                        recompensa = 1
                    elif recompensa == 2:
                        recompensa = -1
                    elif recompensa == -1:
                        recompensa = 0
                    else:
                        recompensa = 0

                    # Calcula el valor esperado:
                    valor = recompensa + gamma * self.estados_valores.get(nuevo_estado, 0)
                    if valor > mejor_valor:  # Si el valor es mejor que el actual:
                        mejor_valor = valor  # Actualiza el mejor valor.
                        mejor_accion = accion  # Actualiza la mejor acción.

                # Actualiza la política si es necesario:
                if self.politica[estado] != mejor_accion:  # Si la acción cambió:
                    politica_estable = False  # La política no es estable.
                    self.politica[estado] = mejor_accion  # Actualiza la política.

        return politica_estable  # Retorna si la política es estable.

    def iteracion_por_politica(self, gamma=0.9, theta=1e-6):
        """
        Realiza la iteración por política hasta que la política sea estable.
        - gamma: Factor de descuento (por defecto 0.9).
        - theta: Umbral de convergencia (por defecto 1e-6).
        """
        self.inicializar_politica()  # Inicializa la política.
        self.inicializar_valores_estado()  # Inicializa los valores de los estados.
        while True:  # Bucle infinito hasta que la política sea estable.
            self.evaluar_politica(gamma, theta)  # Evalúa la política.
            politica_estable = self.mejorar_politica(gamma)  # Mejora la política.
            if politica_estable:  # Si la política es estable:
                break  # Termina el bucle.

    def jugar_con_politica(self):
        """
        Juega una partida utilizando la política óptima aprendida.
        """
        self.juego.tablero = self.juego.crear_tablero()  # Crea un nuevo tablero.
        turno = 1  # Inicializa el turno del jugador 1.
        while self.juego.es_terminal(self.juego.tablero) == 0:  # Mientras el juego no termine:
            estado = self.juego.obtener_estado(self.juego.tablero)  # Obtiene el estado actual.
            if estado in self.politica:  # Si el estado está en la política:
                accion = self.politica[estado]  # Usa la acción de la política.
            else:  # Si no está en la política:
                accion = random.choice(self.juego.posibilidades(self.juego.tablero))  # Elige una acción aleatoria.
            self.juego.tablero = self.juego.aplicar_accion(self.juego.tablero, accion, turno)  # Aplica la acción.
            print(f"Turno del jugador {turno}:\n", self.juego.tablero, "\n")  # Muestra el tablero.
            turno = 3 - turno  # Alterna entre el jugador 1 y 2.
        ganador = self.juego.es_terminal(self.juego.tablero)  # Obtiene el ganador.
        print("¡Gana el jugador!" if ganador in [1, 2] else "Empate.")  # Muestra el resultado.

    def mostrar_valores_y_politica(self, max_mostrar=10):
        """
        Muestra los valores óptimos de los estados y la política óptima.
        - max_mostrar: Número máximo de estados y políticas a mostrar (por defecto 10).
        """
        print("\n🔹 Valores Óptimos de los Estados (V*):")
        for i, (estado, valor) in enumerate(self.estados_valores.items()):  # Recorre los estados y valores.
            print(f"Estado {i+1}: {estado} -> V* = {valor:.3f}")  # Muestra el estado y su valor.
            if i >= max_mostrar - 1:  # Si se alcanza el límite de estados a mostrar:
                break  # Termina el bucle.

        print("\n🔹 Política Óptima (π*):")
        for i, (estado, accion) in enumerate(self.politica.items()):  # Recorre los estados y acciones.
            print(f"Estado {i+1}: {estado} -> Mejor Acción: {accion}")  # Muestra el estado y su mejor acción.
            if i >= max_mostrar - 1:  # Si se alcanza el límite de políticas a mostrar:
                break  # Termina el bucle.