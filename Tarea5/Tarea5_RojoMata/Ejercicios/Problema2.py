import numpy as np
import matplotlib.pyplot as plt

# Clase que define el entorno de control de tráfico en una intersección
class TrafficEnv:
    def __init__(self, max_steps=100, p_arrival=0.3, max_cars_pass=2):
        """
        max_steps: número máximo de pasos por episodio
        p_arrival: probabilidad de que llegue un auto en cada dirección
        max_cars_pass: número máximo de autos que pueden avanzar cuando tienen luz verde
        """
        self.max_steps = max_steps
        self.p_arrival = p_arrival
        self.max_cars_pass = max_cars_pass
        self.reset()

    def reset(self):
        """Reinicia el entorno al estado inicial."""
        self.n_norte = 0  # Autos esperando en dirección Norte-Sur
        self.n_este = 0   # Autos esperando en dirección Este-Oeste
        self.t = 0        # Paso actual del episodio
        self.rewards = []  # Lista de recompensas por paso
        self.states = []   # Historial de estados
        return (self.n_norte, self.n_este)

    def step(self, action):
        """
        Ejecuta un paso del entorno.
        action: 0 = luz verde en dirección Norte-Sur, 1 = luz verde en dirección Este-Oeste
        """

        # Llegada aleatoria de autos a cada dirección
        self.n_norte += np.random.binomial(1, self.p_arrival)
        self.n_este  += np.random.binomial(1, self.p_arrival)

        # Avance de autos según el semáforo
        if action == 0:  # Verde para Norte-Sur
            self.n_norte = max(0, self.n_norte - self.max_cars_pass)
        elif action == 1:  # Verde para Este-Oeste
            self.n_este = max(0, self.n_este - self.max_cars_pass)

        # Recompensa negativa proporcional al total de autos en espera
        reward = - (self.n_norte + self.n_este)
        self.rewards.append(reward)
        self.states.append((self.n_norte, self.n_este))

        # Avanzamos al siguiente paso
        self.t += 1
        done = self.t >= self.max_steps
        return (self.n_norte, self.n_este), reward, done

# ---------------------------
# Simulación de un episodio
# ---------------------------

# Inicialización del entorno
env = TrafficEnv(max_steps=50)
obs = env.reset()
done = False

# Política aleatoria: el agente elige al azar entre NS y EW
while not done:
    action = np.random.choice([0, 1])
    obs, reward, done = env.step(action)

# ---------------------------
# Visualización de recompensas
# ---------------------------

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(env.rewards) + 1), env.rewards, marker='o', linestyle=':', color='magenta')
plt.axhline(0, color='gray', linestyle='-', linewidth=1)
plt.title('Recompensas obtenidas por episodio')
plt.xlabel('Paso')
plt.ylabel('Recompensa')
plt.grid(True)
plt.tight_layout()
plt.savefig("Problema2.png")  # Guarda la imagen como archivo PNG
plt.show()
