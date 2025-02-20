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
        print("🔹 Generando estados aleatorios...")  # Mensaje de inicio.
        estados_generados = set()  # Conjunto para almacenar estados únicos.
        for _ in range(iteraciones):  # Itera el número de veces especificado.
            tablero = self.juego.crear_tablero()  # Crea un nuevo tablero vacío.
            movimientos = random.randint(0, self.juego.tamanio ** 2)  # Número aleatorio de movimientos.
            for _ in range(movimientos):  # Realiza movimientos aleatorios en el tablero.
                posibles_movimientos = self.juego.posibilidades(tablero)  # Obtiene movimientos posibles.
                if not posibles_movimientos:  # Si no hay movimientos posibles:
                    break  # Sale del bucle.
                accion = random.choice(posibles_movimientos)  # Elige una acción aleatoria.
                tablero[accion] = random.choice([1, 2])  # Asigna un jugador (1 o 2) en la posición elegida.

            estado = self.juego.obtener_estado(tablero)  # Obtiene el estado del tablero.
            if self.juego.es_terminal(tablero) == 0 and estado not in estados_generados:  # Si el estado no es terminal y es único:
                estados_generados.add(estado)  # Agrega el estado al conjunto.
                self.ia.estados_valores[estado] = 0  # Inicializa el valor del estado a 0.

        print(f"🔹 Se generaron {len(estados_generados)} estados únicos no terminales.")  # Mensaje de finalización.

    def entrenar_ia(self, metodo="politica", gamma=0.9, theta=1e-6):
        """
        Entrena la IA utilizando el método especificado (iteración por política o por valor).
        - metodo: Método de entrenamiento ("politica" o "valor").
        - gamma: Factor de descuento (por defecto 0.9).
        - theta: Umbral de convergencia (por defecto 1e-6).
        """
        if metodo == "politica":
            print("🔹 Entrenando IA usando iteración por política...")
            self.ia.iteracion_por_politica(gamma, theta)  # Entrena usando iteración por política.
        elif metodo == "valor":
            print("🔹 Entrenando IA usando iteración por valor...")
            self.ia.iteracion_por_valor(gamma, theta)  # Entrena usando iteración por valor.
            self.ia.derivar_politica(gamma)  # Deriva la política óptima a partir de los valores.
        else:
            raise ValueError("Método no válido. Use 'politica' o 'valor'.")

        print("🔹 Entrenamiento completado.")  # Mensaje de finalización.

    def jugar(self, metodo="politica"):
        """
        Juega una partida con la política aprendida.
        - metodo: Método usado para entrenar ("politica" o "valor").
        """
        print(f"\n🔹 Jugando con la política aprendida usando {metodo}...")  # Mensaje de inicio.
        self.ia.jugar_con_politica()  # Llama al método para jugar con la política aprendida.

    def mostrar_resultados(self, metodo="politica", max_mostrar=10):
        """
        Muestra los valores y la política óptima aprendida.
        - metodo: Método usado para entrenar ("politica" o "valor").
        - max_mostrar: Número máximo de estados y políticas a mostrar (por defecto 10).
        """
        print(f"\n🔹 Mostrando resultados del entrenamiento usando {metodo}...")  # Mensaje de inicio.
        self.ia.mostrar_valores_y_politica(max_mostrar)  # Llama al método para mostrar los resultados.


# Ejecutar el programa
if __name__ == "__main__":
    main = Main()  # Crea una instancia de la clase Main.

    # Paso 1: Generar estados aleatorios para el entrenamiento
    main.generar_estados_aleatorios(iteraciones=10000)  # Genera 10,000 estados aleatorios.

    # Paso 2: Entrenar la IA usando iteración por política
    print("\n" + "=" * 50)
    print("🔹 Entrenamiento usando iteración por política 🔹")
    print("=" * 50)
    main.entrenar_ia(metodo="politica")  # Entrena la IA usando iteración por política.
    main.mostrar_resultados(metodo="politica")  # Muestra los resultados de la iteración por política.
    main.jugar(metodo="politica")  # Juega una partida con la política aprendida.

    # Paso 3: Entrenar la IA usando iteración por valor
    print("\n" + "=" * 50)
    print("🔹 Entrenamiento usando iteración por valor 🔹")
    print("=" * 50)
    main.entrenar_ia(metodo="valor")  # Entrena la IA usando iteración por valor.
    main.mostrar_resultados(metodo="valor")  # Muestra los resultados de la iteración por valor.
    main.jugar(metodo="valor")  # Juega una partida con la política aprendida.