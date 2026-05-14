# Importación de librerías necesarias
import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
from collections import defaultdict
import matplotlib.pyplot as plt

# Parámetros del agente
alpha = 0.1          # Tasa de aprendizaje
gamma = 0.99         # Factor de descuento
epsilon = 0.1        # Probabilidad de exploración (epsilon-greedy)
episodios = 1000     # Número total de episodios de entrenamiento

# Inicialización del entorno multiobjetivo
env = mo_gym.make("mo-lunar-lander-v3")
num_act = env.action_space.n  # Número de acciones disponibles

# Se ejecuta una acción aleatoria para determinar la dimensión del vector de recompensa (número de objetivos)
obs, _ = env.reset()
_, r_vec, _, _, _ = env.step(env.action_space.sample())
num_obj = len(r_vec)  # Número de objetivos (dimensión del vector de recompensa)

# Tabla Q que almacena vectores de valores Q para cada par estado-acción
Q = defaultdict(lambda: np.zeros((num_act, num_obj)))

# Lista para registrar las recompensas acumuladas por episodio
r_hist = []

# Conjunto aproximado de soluciones no dominadas (Frontera de Pareto)
front = []

def discret(estado):
    """Discretiza el estado continuo para reducir el espacio de estados y facilitar su almacenamiento."""
    return tuple(np.round(estado, decimals=1))

def acciones_no_dominadas(valores):
    """
    Devuelve una lista de índices de acciones no dominadas según el criterio de Pareto.
    Una acción domina a otra si es al menos igual en todos los objetivos y mejor en al menos uno.
    """
    acciones_nd = []
    for i, vi in enumerate(valores):
        dominado = False
        for j, vj in enumerate(valores):
            if j != i and np.all(vj >= vi) and np.any(vj > vi):
                dominado = True
                break
        if not dominado:
            acciones_nd.append(i)
    return acciones_nd

def selc_accion(estado):
    """
    Selección de acción usando una política epsilon-greedy paretiana.
    Con probabilidad epsilon se elige una acción aleatoria (exploración),
    y con 1 - epsilon se elige una acción no dominada (explotación).
    """
    if np.random.rand() < epsilon:
        return np.random.randint(num_act)
    else:
        valores = Q[estado]
        acciones_nd = acciones_no_dominadas(valores)
        if len(acciones_nd) == 0:
            return np.random.randint(num_act)
        return np.random.choice(acciones_nd)

def actualizar_frontera(frontera, nuevo_vec):
    """
    Actualiza la frontera de Pareto al incorporar un nuevo vector de recompensa.
    Solo se agregan los vectores no dominados y se eliminan los que resulten dominados.
    """
    no_dominados = []
    for vec in frontera:
        if np.all(vec >= nuevo_vec) and np.any(vec > nuevo_vec):
            return frontera  # El nuevo vector está dominado por uno ya existente
        elif not (np.all(nuevo_vec >= vec) and np.any(nuevo_vec > vec)):
            no_dominados.append(vec)  # El vector existente no está dominado por el nuevo
    no_dominados.append(nuevo_vec)
    return no_dominados


# Entrenamiento del agente usando Pareto Q-Learning
for ep in range(episodios):
    estado, _ = env.reset()
    estado = discret(estado)
    terminado = False
    r_acc = np.zeros(num_obj)  # Recompensa acumulada por episodio

    while not terminado:
        accion = selc_accion(estado)
        nv_estado, r_vec, fin, tronco, _ = env.step(accion)
        nv_estado = discret(nv_estado)
        terminado = fin or tronco

        # Cálculo del mejor valor de la siguiente acción
        mejor_sig = np.max(Q[nv_estado], axis=0)

        # Actualización de la tabla Q usando el valor máximo paretiano
        Q[estado][accion] = (1 - alpha) * Q[estado][accion] + alpha * (r_vec + gamma * mejor_sig)

        # Actualización del estado y acumulación de recompensas
        estado = nv_estado
        r_acc += r_vec

    # Guardar recompensa acumulada del episodio
    r_hist.append(r_acc)

    # Actualizar la frontera de Pareto con la nueva recompensa
    front = actualizar_frontera(front, r_acc)

    # Imprimir resultados cada 50 episodios, formateado como tabla
    if ep % 50 == 0:
        r_redondeada = np.round(r_acc, 2)
        print(f"{'='*60}")
        print(f"Episodio {ep}")
        print(f"{'Objetivo':<10}{'Valor':>10}")
        for i, val in enumerate(r_redondeada):
            print(f"{'Obj ' + str(i+1):<10}{val:>10}")

# Cierre del entorno
env.close()

# Gráfica de recompensas acumuladas por episodio
r_dicc = np.array(r_hist)
plt.plot(r_dicc[:, 0], label='Recompensa 1')
plt.plot(r_dicc[:, 1], label='Recompensa 2')
plt.title("Recompensas por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("RecompensasQ.png")
plt.show()

# Gráfica de la Frontera de Pareto aproximada
front = np.array(front)
plt.scatter(front[:, 0], front[:, 1], color='red')
plt.title("Frontera de Pareto aproximada")
plt.xlabel("Objetivo 1")
plt.ylabel("Objetivo 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("Frente de Pareto Q.png")
plt.show()
