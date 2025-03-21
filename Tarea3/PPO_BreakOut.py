import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from minatar import Environment  # Librería MinAtar para el juego

# ==============================
# 🔹 CONFIGURACIÓN DEL ENTORNO
# ==============================
entorno_juego = Environment("breakout")
num_acciones_posibles = entorno_juego.num_actions()
forma_estado_juego = entorno_juego.state_shape()  # (10, 10, 4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usar GPU si está disponible

# ==============================
# 🔹 DEFINICIÓN DE MODELOS
# ==============================

class Actor(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * input_shape[0] * input_shape[1], 256)
        self.fc2 = nn.Linear(256, num_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.softmax(self.fc2(x))


class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * input_shape[0] * input_shape[1], 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ==============================
# 🔹 MEMORIA PARA PPO
# ==============================
class Memory:
    def __init__(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones = [], [], [], [], []

    def store(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())  # 🔹 Desconectar gradientes
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones = [], [], [], [], []

# ==============================
# 🔹 FUNCIONES AUXILIARES
# ==============================

def compute_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros(len(rewards))
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage

    return advantages

def update(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

# ==============================
# 🔹 ENTRENAMIENTO PPO
# ==============================

# Inicializar redes y optimizadores
actor = Actor(forma_estado_juego, num_acciones_posibles).to(device)
critic = Critic(forma_estado_juego).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
memory = Memory()

recompensas_historicas = []
K_epochs = 4

for episode in range(1000):
    entorno_juego.reset()
    done = False
    recompensa_total = 0

    while not done:
        estado = torch.tensor(entorno_juego.state(), dtype=torch.float32).to(device)
        estado = estado.permute(2, 0, 1).unsqueeze(0)

        action_probs = actor(estado)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        reward, done = entorno_juego.act(action.item())
        recompensa_total += reward

        memory.store(estado, action, log_prob, reward, done)

    recompensas_historicas.append(recompensa_total)

    # Obtener valores de estados
    states_tensor = torch.cat(memory.states).to(device)
    values = critic(states_tensor).squeeze().detach().cpu().numpy()
    values = np.append(values, 0)

    # Calcular ventajas
    advantages = torch.tensor(compute_advantages(memory.rewards, values, memory.dones), dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Calcular returns
    returns = []
    discounted_sum = 0
    for r, done in zip(reversed(memory.rewards), reversed(memory.dones)):
        discounted_sum = r + 0.99 * discounted_sum * (1 - done)
        returns.insert(0, discounted_sum)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    # Evitar que old_log_probs esté conectado al grafo computacional
    old_log_probs = torch.cat(memory.log_probs).detach().to(device)

    for _ in range(K_epochs):
        # 🔹 Recalcular nuevas probabilidades de acción en cada iteración
        action_probs = actor(states_tensor)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(torch.stack(memory.actions).to(device))

        update(actor_optimizer, ppo_loss(old_log_probs, new_log_probs, advantages))
        update(critic_optimizer, ((returns - critic(states_tensor).squeeze()) ** 2).mean())

    memory.clear()

# ==============================
# 🔹 VISUALIZACIÓN DEL ENTRENAMIENTO
# ==============================
plt.plot(recompensas_historicas, label="Recompensa por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.legend()
plt.show()
