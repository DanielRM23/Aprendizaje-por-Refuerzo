# Importamos las librerías necesarias
import gymnasium as gym           # Entorno de simulación RL
import numpy as np                # Operaciones numéricas
import matplotlib.pyplot as plt   # Visualización de resultados
import random                     # Para la política epsilon-greedy

# Crear el entorno FrozenLake sin efecto de resbalón (determinista)
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n    # Número total de estados
n_actions = env.action_space.n        # Número total de acciones posibles

# Definición de la política epsilon-greedy clásica
def politica_egreedy(Q, state, epsilon):
    """
    Selecciona una acción con probabilidad epsilon (exploración aleatoria),
    o la mejor acción conocida hasta el momento (explotación).
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, n_actions - 1)  # Acción aleatoria
    return np.argmax(Q[state])                  # Acción con mayor valor Q

# Reducción de epsilon a lo largo del entrenamiento
def actualizar_epsilon(epsilon, min_epsilon, tasa_reduccion):
    """
    Reduce epsilon de forma exponencial hasta un valor mínimo.
    """
    return max(min_epsilon, epsilon * (1 - tasa_reduccion))

# Entrenamiento con Q-learning clásico y decaimiento de epsilon
def entrenar_qlearning_clasico(n_episodes, tasa_reduccion=0.001):
    Q = np.zeros((n_states, n_actions))  # Tabla Q inicializada en ceros
    alpha = 0.1                          # Tasa de aprendizaje
    gamma = 0.99                         # Factor de descuento
    epsilon = 1.0                        # Exploración inicial
    min_epsilon = 0.01                  # Exploración mínima
    rewards = []                        # Recompensas por episodio

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Seleccionar acción usando política epsilon-greedy clásica
            action = politica_egreedy(Q, state, epsilon)

            # Ejecutar acción y observar resultado
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Actualizar la tabla Q con la fórmula Q-learning
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

        # Actualizar epsilon después de cada episodio
        epsilon = actualizar_epsilon(epsilon, min_epsilon, tasa_reduccion)
        rewards.append(total_reward)

    # Calcular recompensas promedio por bloques de 100 episodios
    avg_rewards = [np.mean(rewards[i:i+100]) for i in range(0, len(rewards), 100)]
    return avg_rewards

# Entrenamiento con Q-learning y política guiada por confianza (novedosa)
def entrenar_qlearning_confianza(n_episodes, c=2):
    Q = np.zeros((n_states, n_actions))  # Tabla Q
    N = np.zeros(n_states)               # Contador de visitas por estado
    alpha = 0.1                          # Tasa de aprendizaje
    gamma = 0.99                         # Factor de descuento
    rewards = []                         # Recompensas por episodio

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Calcular epsilon en función de la "confianza" (visitas)
            epsilon = min(1.0, c / np.sqrt(N[state] + 1))

            # Seleccionar acción usando política adaptativa por estado
            action = politica_egreedy(Q, state, epsilon)

            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Actualización Q-learning
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            # Actualizar contador de visitas para ese estado
            N[state] += 1
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    # Calcular recompensas promedio por bloques de 100 episodios
    avg_rewards = [np.mean(rewards[i:i+100]) for i in range(0, len(rewards), 100)]
    return avg_rewards

# Número total de episodios de entrenamiento
n_episodes = 10_000

# Entrenamiento de ambos métodos
rewards_decay = entrenar_qlearning_clasico(n_episodes)       # Clásico con epsilon decreciente
rewards_confianza = entrenar_qlearning_confianza(n_episodes) # Novedoso con confianza

# Graficar comparación entre ambos métodos
plt.figure(figsize=(10,6))
plt.plot(rewards_decay, label="Q-Learning con ε decreciente")
plt.plot(rewards_confianza, label="Exploración guiada por confianza")
plt.xlabel("Bloques de 100 episodios")
plt.ylabel("Recompensa promedio")
plt.title("Comparativa de rendimiento")
plt.grid()
plt.legend()
plt.savefig("Comparativa_Novedoso_vs_Decaimiento.png")  # Guardar imagen para el reporte
