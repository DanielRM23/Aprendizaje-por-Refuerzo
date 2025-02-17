import random  
import numpy as np 
import MDP_Gato  # Importa el módulo MDP_Gato que contiene la clase TicTacToe.
import Iteracion_de_politica  # Importa el módulo Iteracion_de_politica que contiene la clase TicTacToeAI.

class Main:
    def __init__(self):
        """
        Inicializa la clase Main.
        - Crea una instancia del juego TicTacToe desde el módulo MDP_Gato.
        - Crea una instancia de la IA TicTacToeAI desde el módulo Iteracion_de_politica.
        """
        self.juego = MDP_Gato.TicTacToe()  # Instancia del juego TicTacToe.
        self.ia = Iteracion_de_politica.TicTacToeAI(self.juego)  # Instancia de la IA, pasando el juego como argumento.

    def generar_estados_aleatorios(self, iteraciones=10000):
        """
        Genera estados aleatorios para entrenar la IA.
        - iteraciones: Número de estados aleatorios a generar (por defecto 10,000).
        Evita generar estados terminales, ya que no aportan información útil.
        """
        #print("🔹 Generando estados aleatorios...")  # Opcional: Mensaje de inicio.
        estados_generados = set()  # Conjunto para almacenar estados únicos.
        for iterador1 in range(iteraciones):  # Itera el número de veces especificado.
            tablero = self.juego.crear_tablero()  # Crea un nuevo tablero vacío.
            movimientos = random.randint(0, self.juego.tamanio ** 2)  # Número aleatorio de movimientos.
            for iterador2 in range(movimientos):  # Realiza movimientos aleatorios en el tablero.
                posibles_movimientos = self.juego.posibilidades(tablero)  # Obtiene movimientos posibles.
                if not posibles_movimientos:  # Si no hay movimientos posibles:
                    break  # Sale del bucle.
                accion = random.choice(posibles_movimientos)  # Elige una acción aleatoria.
                tablero[accion] = random.choice([1, 2])  # Asigna un jugador (1 o 2) en la posición elegida.

            estado = self.juego.obtener_estado(tablero)  # Obtiene el estado del tablero.
            if self.juego.es_terminal(tablero) == 0 and estado not in estados_generados:  # Si el estado no es terminal y es único:
                estados_generados.add(estado)  # Agrega el estado al conjunto.
                self.ia.estados_valores[estado] = 0  # Inicializa el valor del estado a 0.

        #print(f"🔹 Se generaron {len(estados_generados)} estados únicos no terminales.")  # Opcional: Mensaje de finalización.

    def entrenar_ia(self):
        """
        Entrena la IA utilizando iteración por política.
        """
        #print("🔹 Entrenando agente...")  # Opcional: Mensaje de inicio.
        self.ia.iteracion_por_politica()  # Llama al método de iteración por política.
        #print("🔹 Entrenamiento completado.")  # Opcional: Mensaje de finalización.

    def jugar(self):
        """
        Juega una partida con la política aprendida.
        """
        print("\n🔹 Jugando con la política aprendida...")  # Mensaje de inicio.
        self.ia.jugar_con_politica()  # Llama al método para jugar con la política aprendida.

    def mostrar_resultados(self):
        """
        Muestra los valores y la política óptima aprendida.
        """
        # print("\n🔹 Mostrando resultados del entrenamiento...")  # Opcional: Mensaje de inicio.
        self.ia.mostrar_valores_y_politica()  # Llama al método para mostrar los resultados.


# Ejecutar el programa
if __name__ == "__main__":
    main = Main()  # Crea una instancia de la clase Main.

    # Generar estados aleatorios para el entrenamiento
    main.generar_estados_aleatorios(iteraciones=10000)  # Genera 10,000 estados aleatorios.

    # Entrenar al agente
    main.entrenar_ia()  # Entrena la IA usando iteración por política.

    #Mostrar los resultados del entrenamiento
    main.mostrar_resultados()  # Muestra los valores y la política óptima.

    # Jugar una partida con la política aprendida
    main.jugar()  # Juega una partida usando la política aprendida.