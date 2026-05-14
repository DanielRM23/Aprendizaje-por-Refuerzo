# ---------------------------------------
# Librerías necesarias
# ---------------------------------------
import gym                              # Librería base de entornos de RL
from gym import spaces                  # Espacios de acción y observación
import numpy as np                      # Operaciones numéricas y aleatorias
import matplotlib.pyplot as plt         # Visualización de resultados
from stable_baselines3 import PPO       # Algoritmo PPO de la librería SB3

# ---------------------------------------
# Definición del entorno personalizado
# ---------------------------------------
class Extra4(gym.Env):
    """
    Entorno tipo Bandido Estocástico:
    - Las acciones no tienen efecto en la recompensa.
    - Las recompensas son +1 o -1, generadas aleatoriamente con probabilidad uniforme.
    """
    def __init__(self):
        super(Extra4, self).__init__()

        # Espacio de acción: 2 posibles acciones (0 o 1)
        self.action_space = spaces.Discrete(2)

        # Espacio de observación: un valor constante (dummy), siempre 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Estado interno (no tiene relevancia en este entorno)
        self.estado = 0

    def reset(self, seed=None, options=None):
        # Reinicia el entorno; siempre regresa el mismo estado
        super().reset(seed=seed)
        self.estado = 0
        observacion = np.array([self.estado], dtype=np.float32)
        return observacion, {}

    def step(self, accion):
        # Ignora la acción y devuelve una recompensa aleatoria: -1 o +1
        recompensa = np.random.choice([-1, 1])
        terminado = False  # El entorno nunca termina (no episódico)
        observacion = np.array([self.estado], dtype=np.float32)
        return observacion, recompensa, terminado, False, {}

    def render(self, mode='human'):
        # No se implementa visualización
        pass

# ---------------------------------------
# Entrenamiento con Q-learning clásico
# ---------------------------------------
def entrenar_con_qlearning():
    entorno = Extra4()                                      # Instancia del entorno
    tabla_q = np.zeros(entorno.action_space.n)              # Tabla Q inicializada en ceros
    tasa_aprendizaje = 0.1                                  # Alpha: velocidad de aprendizaje
    factor_descuento = 0.95                                 # Gamma: importancia de recompensas futuras
    epsilon = 0.1                                            # Probabilidad de explorar en vez de explotar

    total_episodios = 1000
    recompensas = []                                        # Recompensas acumuladas por episodio

    for episodio in range(total_episodios):
        estado, _ = entorno.reset()
        recompensa_total = 0

        # Solo se realiza un paso por episodio (como en problemas de bandido)
        for _ in range(1):
            if np.random.rand() < epsilon:
                accion = entorno.action_space.sample()      # Acción aleatoria (exploración)
            else:
                accion = np.argmax(tabla_q)                 # Mejor acción (explotación)

            _, recompensa, _, _, _ = entorno.step(accion)

            # Actualización de la Q-table con la ecuación de Bellman
            tabla_q[accion] += tasa_aprendizaje * (recompensa + factor_descuento * np.max(tabla_q) - tabla_q[accion])
            recompensa_total += recompensa

        recompensas.append(recompensa_total)

    # Suavizado con promedio móvil
    ventana = 50
    promedio_movil = np.convolve(recompensas, np.ones(ventana)/ventana, mode='valid')

    # Gráfica de resultados
    plt.figure(figsize=(10, 5))
    plt.plot(promedio_movil, color="red")
    plt.title("Q-learning")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Promedio")
    plt.grid()
    plt.savefig("QLearning_Extra4.png")
    plt.show()

# ---------------------------------------
# Entrenamiento con PPO (Stable-Baselines3)
# ---------------------------------------
def entrenar_con_ppo():
    entorno = Extra4()                                      # Instancia del entorno
    modelo = PPO("MlpPolicy", entorno, verbose=0)           # Inicializa PPO con red MLP
    modelo.learn(total_timesteps=10000)                     # Entrena el modelo

    observacion, _ = entorno.reset()
    recompensas = []                                        # Recompensas acumuladas por paso

    for _ in range(1000):
        accion, _ = modelo.predict(observacion)             # Predice acción usando la política entrenada
        observacion, recompensa, terminado, truncado, _ = entorno.step(accion)
        recompensas.append(recompensa)

        if terminado or truncado:
            observacion, _ = entorno.reset()

    # Suavizado con promedio móvil
    ventana = 50
    promedio_movil = np.convolve(recompensas, np.ones(ventana)/ventana, mode='valid')

    # Gráfica de resultados
    plt.figure(figsize=(10, 5))
    plt.plot(promedio_movil, color="magenta")
    plt.title("PPO")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Promedio")
    plt.grid()
    plt.savefig("PPO_Extra.png")
    plt.show()

# ---------------------------------------
# Punto de entrada principal
# ---------------------------------------
if __name__ == "__main__":
    print("Entrenando con PPO...")
    entrenar_con_ppo()

    print("Entrenando con Q-learning...")
    entrenar_con_qlearning()
