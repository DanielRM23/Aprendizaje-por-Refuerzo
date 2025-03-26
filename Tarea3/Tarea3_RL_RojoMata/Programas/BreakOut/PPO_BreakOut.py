# ==============================================
# PPO aplicado al entorno breakout
# ==============================================


# ===================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ===================================================================
import numpy as np  # Librería para manejo de arreglos y funciones matemáticas
import random  # Para selección aleatoria
import torch  # Librería principal para computación en GPU y tensores
import torch.nn as nn  # Módulos para redes neuronales
import torch.optim as optim  # Métodos de optimización como Adam
from collections import deque  # Cola doblemente terminada, útil para el replay buffer
from minatar import Environment  # Entorno MinAtar para el juego Breakout
import matplotlib.pyplot as plt  # Para graficar resultados
import time  # Para pausas temporales

# ==============================
# CONFIGURACIÓN DEL ENTORNO
# ==============================
env = Environment("breakout")  # Crear entorno Breakout
num_acciones = env.num_actions()  # Número total de acciones posibles
tamano_estado = env.state_shape()  # Dimensión del estado del entorno
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Seleccionar GPU si está disponible

# ==============================
# DEFINICIÓN DE MODELOS
# ==============================
class RedActor(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(RedActor, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, 3, 1, 1)  # Capa convolucional 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)  # Capa convolucional 2
        self.fc1 = nn.Linear(64 * input_shape[0] * input_shape[1], 256)  # Capa totalmente conectada
        self.fc2 = nn.Linear(256, num_actions)  # Capa de salida: logits para cada acción
        self.softmax = nn.Softmax(dim=-1)  # Convertir logits a probabilidades

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Aplicar ReLU a la salida de conv1
        x = torch.relu(self.conv2(x))  # Aplicar ReLU a la salida de conv2
        x = x.reshape(x.size(0), -1)  # Aplanar el tensor
        x = torch.relu(self.fc1(x))  # ReLU sobre capa densa
        return self.softmax(self.fc2(x))  # Devolver distribución de acciones

class RedCritica(nn.Module):
    def __init__(self, input_shape):
        super(RedCritica, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, 3, 1, 1)  # Capa convolucional 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)  # Capa convolucional 2
        self.fc1 = nn.Linear(64 * input_shape[0] * input_shape[1], 256)  # Capa totalmente conectada
        self.fc2 = nn.Linear(256, 1)  # Capa de salida: valor escalar del estado

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # ReLU sobre conv1
        x = torch.relu(self.conv2(x))  # ReLU sobre conv2
        x = x.reshape(x.size(0), -1)  # Aplanar
        x = torch.relu(self.fc1(x))  # ReLU sobre capa densa
        return self.fc2(x)  # Devolver valor del estado

# ==============================
# MEMORIA PARA PPO
# ==============================

class MemoriaPPO:
    def __init__(self):
        self.estados = []  # Lista para estados
        self.acciones = []  # Lista para acciones tomadas
        self.log_probs = []  # Lista para log-probs de acciones
        self.recompensas = []  # Lista de recompensas
        self.dones = []  # Lista de flags de fin de episodio

    def guardar(self, estado, accion, log_prob, recompensa, terminado):
        self.estados.append(estado)
        self.acciones.append(accion)
        self.log_probs.append(log_prob.detach())  # Desconectar de la red
        self.recompensas.append(recompensa)
        self.dones.append(terminado)

    def limpiar(self):
        self.estados, self.acciones, self.log_probs, self.recompensas, self.dones = [], [], [], [], []

# ==============================
# FUNCIONES AUXILIARES
# ==============================

def calcular_ventajas(recompensas, valores, dones, gamma=0.99, lam=0.95):
    ventajas = np.zeros(len(recompensas))  # Inicializar ventajas
    ventaja_acumulada = 0  # GAE acumulada
    for t in reversed(range(len(recompensas))):
        delta = recompensas[t] + gamma * valores[t + 1] * (1 - dones[t]) - valores[t]
        ventajas[t] = ventaja_acumulada = delta + gamma * lam * (1 - dones[t]) * ventaja_acumulada
    return ventajas

def actualizar(optimizador, perdida):
    optimizador.zero_grad()  # Limpiar gradientes
    perdida.backward()  # Retropropagación
    optimizador.step()  # Paso de optimización

def funcion_perdida_ppo(logs_antiguos, logs_nuevos, ventajas, epsilon=0.2):
    ratio = torch.exp(logs_nuevos - logs_antiguos)  # Cálculo del ratio
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)  # Aplicar clipping
    return -torch.min(ratio * ventajas, clipped * ventajas).mean()  # PPO loss

# ==============================
# ENTRENAMIENTO PPO
# ==============================

actor = RedActor(tamano_estado, num_acciones).to(dispositivo)  # Inicializar actor
critico = RedCritica(tamano_estado).to(dispositivo)  # Inicializar crítico
opt_actor = optim.Adam(actor.parameters(), lr=1e-4)  # Optimizador del actor
opt_critico = optim.Adam(critico.parameters(), lr=1e-4)  # Optimizador del crítico
memoria = MemoriaPPO()  # Instancia de memoria
recompensas_totales = []  # Recompensas por episodio
K_epocas = 4  # Número de actualizaciones por episodio

for episodio in range(1000):
    env.reset()  # Reiniciar entorno
    terminado = False
    recompensa_episodio = 0  # Acumulador de recompensa

    while not terminado:
        estado_bruto = torch.tensor(env.state(), dtype=torch.float32).to(dispositivo)  # Obtener estado
        estado = estado_bruto.permute(2, 0, 1).unsqueeze(0)  # Reordenar a [batch, canales, alto, ancho]

        probs = actor(estado)  # Obtener distribución de acción
        dist = torch.distributions.Categorical(probs)
        accion = dist.sample()  # Muestrear acción
        log_prob = dist.log_prob(accion)  # Log-probabilidad

        recompensa, terminado = env.act(accion.item())  # Ejecutar acción
        recompensa_episodio += recompensa  # Acumular recompensa

        memoria.guardar(estado, accion, log_prob, recompensa, terminado)  # Guardar transición

    recompensas_totales.append(recompensa_episodio)  # Registrar recompensa del episodio

    batch_estados = torch.cat(memoria.estados).to(dispositivo)  # Unir todos los estados
    valores = critico(batch_estados).squeeze().detach().cpu().numpy()  # Obtener valores V(s)
    valores = np.append(valores, 0)  # Añadir V(s') = 0 para último paso

    ventajas = torch.tensor(calcular_ventajas(memoria.recompensas, valores, memoria.dones), dtype=torch.float32).to(dispositivo)
    ventajas = (ventajas - ventajas.mean()) / (ventajas.std() + 1e-8)  # Normalizar

    retornos = []
    descuento = 0
    for r, d in zip(reversed(memoria.recompensas), reversed(memoria.dones)):
        descuento = r + 0.99 * descuento * (1 - d)
        retornos.insert(0, descuento)
    retornos = torch.tensor(retornos, dtype=torch.float32).to(dispositivo)

    logs_antiguos = torch.cat(memoria.log_probs).to(dispositivo)  # Guardar log_probs antiguos
    acciones = torch.tensor(memoria.acciones, dtype=torch.long).to(dispositivo)  # Acciones tomadas

    for _ in range(K_epocas):
        probs = actor(batch_estados)  # Recalcular probas
        dist = torch.distributions.Categorical(probs)
        logs_nuevos = dist.log_prob(acciones)  # Nuevas log_probs

        perdida_actor = funcion_perdida_ppo(logs_antiguos, logs_nuevos, ventajas)
        perdida_critico = ((retornos - critico(batch_estados).squeeze()) ** 2).mean()

        actualizar(opt_actor, perdida_actor)  # Actualizar actor
        actualizar(opt_critico, perdida_critico)  # Actualizar crítico

    memoria.limpiar()  # Limpiar memoria tras cada episodio

# ==============================
# VISUALIZACIÓN DEL ENTRENAMIENTO
# ==============================

plt.plot(recompensas_totales, label="Recompensa por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("PPO en MinAtar Breakout")
plt.legend()
plt.savefig("PPO_BreakOut.png")
plt.show()
