# ==============================================
# DDQN aplicado al entorno ALE/Pong-v5
# ==============================================


# ------------------------------------------------------
# Importaciones necesarias
# ------------------------------------------------------

import gymnasium as gym  # Entorno de simulación
import numpy as np  # Cálculo numérico
import random  # Generación de números aleatorios
import torch  # Operaciones con tensores (CPU y GPU)
import torch.nn as nn  # Módulos de redes neuronales
import torch.optim as optim  # Optimizadores
from collections import deque  # Estructura de datos tipo cola
import matplotlib.pyplot as plt  # Visualización de gráficos
import cv2  # Procesamiento de imágenes
import time  # Medición de tiempo
import ale_py  # Interfaz de Atari Learning Environment (usado por Gym)

# ------------------------------------------------------
# Función de preprocesamiento de imágenes
# ------------------------------------------------------
def preprocesar(observacion):
    """Convierte una imagen RGB en escala de grises, la redimensiona a 84x84 y normaliza los valores.

    Args:
        observacion (ndarray): Imagen del entorno en RGB.

    Returns:
        ndarray: Imagen preprocesada en escala de grises y normalizada.
    """
    gris = cv2.cvtColor(observacion, cv2.COLOR_RGB2GRAY)  # Convertir a escala de grises
    redimensionada = cv2.resize(gris, (84, 84), interpolation=cv2.INTER_AREA)  # Redimensionar
    normalizada = redimensionada.astype(np.float32) / 255.0  # Normalizar entre 0 y 1
    return normalizada

# ------------------------------------------------------
# Definición de la red neuronal Q
# ------------------------------------------------------
class RedQ(nn.Module):
    def __init__(self, num_acciones):
        """Red neuronal convolucional para estimar valores Q.

        Args:
            num_acciones (int): Número de acciones posibles en el entorno.
        """
        super(RedQ, self).__init__()  # Inicializa clase padre nn.Module
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # Capa convolucional 1
        self.relu1 = nn.ReLU()  # Activación ReLU 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # Capa convolucional 2
        self.relu2 = nn.ReLU()  # Activación ReLU 2
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # Capa convolucional 3
        self.relu3 = nn.ReLU()  # Activación ReLU 3
        self.flatten = nn.Flatten()  # Aplanado para entrada a capa densa
        self.fc1 = nn.Linear(7 * 7 * 64, 512)  # Capa densa totalmente conectada
        self.relu4 = nn.ReLU()  # Activación ReLU 4
        self.out = nn.Linear(512, num_acciones)  # Capa de salida para valores Q

    def forward(self, x):
        """Propagación hacia adelante de la red.

        Args:
            x (Tensor): Entrada con forma (batch_size, 4, 84, 84).

        Returns:
            Tensor: Valores Q para cada acción.
        """
        x = self.relu1(self.conv1(x))  # Paso por capa 1 y ReLU
        x = self.relu2(self.conv2(x))  # Paso por capa 2 y ReLU
        x = self.relu3(self.conv3(x))  # Paso por capa 3 y ReLU
        x = self.flatten(x)  # Aplana la salida
        x = self.relu4(self.fc1(x))  # Capa densa + ReLU
        return self.out(x)  # Valores Q

# ------------------------------------------------------
# Buffer de experiencia (Replay Memory)
# ------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacidad):
        """Inicializa el buffer de experiencia.

        Args:
            capacidad (int): Tamaño máximo del buffer.
        """
        self.buffer = deque(maxlen=capacidad)  # Cola de tamaño fijo

    def agregar(self, estado, accion, recompensa, siguiente_estado, terminado):
        """Agrega una transición al buffer.

        Args:
            estado: Estado actual.
            accion: Acción tomada.
            recompensa: Recompensa obtenida.
            siguiente_estado: Estado siguiente.
            terminado: Si el episodio terminó.
        """
        self.buffer.append((estado, accion, recompensa, siguiente_estado, terminado))  # Agrega transición

    def sample(self, batch_size):
        """Devuelve una muestra aleatoria de transiciones.

        Args:
            batch_size (int): Número de muestras a devolver.

        Returns:
            Tuple: Lote de transiciones.
        """
        batch = random.sample(self.buffer, batch_size)  # Muestra aleatoria
        estados, acciones, recompensas, siguientes, terminales = zip(*batch)  # Desempaquetar
        return (np.array(estados), acciones, recompensas, np.array(siguientes), terminales)

    def __len__(self):
        return len(self.buffer)  # Tamaño actual del buffer

# ------------------------------------------------------
# Política epsilon-greedy para exploración/explotación
# ------------------------------------------------------
class PoliticaEpsilonGreedy:
    def __init__(self, num_acciones, epsilon=1.0, epsilon_min=0.01, decay=0.995):
        """Inicializa la política epsilon-greedy.

        Args:
            num_acciones (int): Número de acciones disponibles.
            epsilon (float): Valor inicial de epsilon.
            epsilon_min (float): Valor mínimo de epsilon.
            decay (float): Tasa de decaimiento de epsilon.
        """
        self.epsilon = epsilon  # Probabilidad de exploración
        self.epsilon_min = epsilon_min  # Límite inferior de epsilon
        self.decay = decay  # Factor de decaimiento
        self.num_acciones = num_acciones  # Número total de acciones

    def seleccionar_accion(self, estado, red_q, dispositivo):
        """Selecciona una acción usando epsilon-greedy.

        Args:
            estado (ndarray): Estado actual.
            red_q (RedQ): Red Q para estimar valores.
            dispositivo (torch.device): CPU o GPU.

        Returns:
            int: Acción seleccionada.
        """
        if random.random() < self.epsilon:  # Exploración
            return random.randint(0, self.num_acciones - 1)
        with torch.no_grad():  # Desactiva el cálculo de gradientes
            estado_tensor = torch.tensor(np.array([estado]), dtype=torch.float32, device=dispositivo)
            q_values = red_q(estado_tensor)  # Estimar valores Q
        return torch.argmax(q_values).item()  # Acción con valor Q máximo

    def decaer(self):
        """Reduce el valor de epsilon de forma exponencial."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)  # Aplica decaimiento

# ------------------------------------------------------
# Inicialización del entorno y modelos
# ------------------------------------------------------
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Selecciona GPU si está disponible
print("¿CUDA disponible?", torch.cuda.is_available())  # Imprime disponibilidad de GPU
print("Usando dispositivo:", DISPOSITIVO)  # Muestra dispositivo usado

env = gym.make("ALE/Pong-v5")  # Crea el entorno Pong
obs, info = env.reset()  # Reinicia el entorno y obtiene observación inicial
action_space = env.action_space.n  # Número de acciones disponibles

# Hiperparámetros
GAMMA = 0.99  # Factor de descuento para recompensa futura
LR = 1e-4  # Tasa de aprendizaje del optimizador
BATCH_SIZE = 32  # Tamaño del minibatch para entrenamiento
BUFFER_SIZE = 100_000  # Capacidad máxima del buffer de experiencia
TARGET_UPDATE = 1000  # Frecuencia para actualizar la red objetivo
EPISODES = 500  # Número total de episodios de entrenamiento
ENTRENAR_CADA = 4  # Entrenar cada 4 pasos
WARMUP_STEPS = 1000  # Esperar estos pasos antes de entrenar

# Redes neuronales
red_q = RedQ(action_space).to(DISPOSITIVO)  # Red Q principal
red_obj = RedQ(action_space).to(DISPOSITIVO)  # Red Q objetivo (target)
red_obj.load_state_dict(red_q.state_dict())  # Copia pesos de la red principal a la red objetivo
red_obj.eval()  # Desactiva el modo de entrenamiento en la red objetivo
optimizador = optim.Adam(red_q.parameters(), lr=LR)  # Inicializa el optimizador Adam
replay = ReplayBuffer(BUFFER_SIZE)  # Inicializa el buffer de experiencia
politica = PoliticaEpsilonGreedy(action_space)  # Inicializa la política epsilon-greedy

# ------------------------------------------------------
# Función de entrenamiento
# ------------------------------------------------------
def entrenar():
    """Entrena la red principal utilizando muestras del buffer."""
    if len(replay) < BATCH_SIZE:  # Si no hay suficientes datos, salir
        return
    estados, acciones, recompensas, siguientes, terminales = replay.sample(BATCH_SIZE)  # Obtener minibatch

    # Convertir a tensores y mover al dispositivo
    estados = torch.tensor(estados, dtype=torch.float32, device=DISPOSITIVO)
    acciones = torch.tensor(acciones, dtype=torch.int64, device=DISPOSITIVO).unsqueeze(1)
    recompensas = torch.tensor(recompensas, dtype=torch.float32, device=DISPOSITIVO).unsqueeze(1)
    siguientes = torch.tensor(siguientes, dtype=torch.float32, device=DISPOSITIVO)
    terminales = torch.tensor(terminales, dtype=torch.float32, device=DISPOSITIVO).unsqueeze(1)

    q_actual = red_q(estados).gather(1, acciones)  # Q(s, a) de la red actual
    with torch.no_grad():  # No calcular gradientes para la red objetivo
        acciones_max = red_q(siguientes).argmax(1, keepdim=True)  # Acción con Q máximo
        q_siguiente = red_obj(siguientes).gather(1, acciones_max)  # Q(s', argmax_a' Q(s',a'))
        q_objetivo = recompensas + GAMMA * q_siguiente * (1 - terminales)  # Target Q-value

    perdida = nn.MSELoss()(q_actual, q_objetivo)  # Calcula la pérdida MSE
    optimizador.zero_grad()  # Reinicia los gradientes
    perdida.backward()  # Calcula gradientes
    optimizador.step()  # Actualiza los pesos de la red principal

# ------------------------------------------------------
# Loop principal de entrenamiento
# ------------------------------------------------------
recompensas = []  # Lista para guardar recompensa por episodio
frames_stack = deque(maxlen=4)  # Cola para almacenar 4 frames consecutivos
pasos = 0  # Contador global de pasos

for ep in range(EPISODES):  # Bucle sobre los episodios
    inicio = time.time()  # Marca inicio del episodio
    obs, _ = env.reset()  # Reinicia el entorno
    frame = preprocesar(obs)  # Preprocesa la observación inicial
    frames_stack.clear()  # Limpia la pila de frames
    for _ in range(4):  # Llena pila inicial con 4 frames iguales
        frames_stack.append(frame)
    estado = np.stack(frames_stack, axis=0)  # Forma estado inicial (4, 84, 84)

    total = 0  # Acumulador de recompensa del episodio
    terminado = False  # Bandera de finalización
    while not terminado:  # Loop de pasos en el episodio
        accion = politica.seleccionar_accion(estado, red_q, DISPOSITIVO)  # Elegir acción con política epsilon-greedy
        obs_, recompensa, termin, trunc, _ = env.step(accion)  # Ejecuta acción en el entorno
        terminado = termin or trunc  # Verifica si terminó el episodio
        frame_ = preprocesar(obs_)  # Preprocesa nueva observación
        frames_stack.append(frame_)  # Agrega a la pila
        siguiente_estado = np.stack(frames_stack, axis=0)  # Forma nuevo estado

        replay.agregar(estado, accion, recompensa, siguiente_estado, terminado)  # Guarda transición en buffer

        if pasos > WARMUP_STEPS and pasos % ENTRENAR_CADA == 0:  # Entrenar si ya pasamos el warmup y cada cierto número de pasos
            entrenar()

        estado = siguiente_estado  # Actualiza el estado actual
        total += recompensa  # Suma recompensa del paso
        pasos += 1  # Aumenta contador global

        if pasos % TARGET_UPDATE == 0:  # Cada cierto número de pasos, actualiza la red objetivo
            red_obj.load_state_dict(red_q.state_dict())

    politica.decaer()  # Reduce epsilon
    recompensas.append(total)  # Guarda recompensa del episodio
    duracion = time.time() - inicio  # Tiempo de ejecución del episodio
    print(f"Episodio {ep+1}/{EPISODES} | Recompensa: {total:.2f} | Duración: {duracion:.2f}s | Epsilon: {politica.epsilon:.4f}")  # Info del episodio


# ------------------------------------------------------
# Gráfica
# ------------------------------------------------------
def media_movil(lista, ventana=20):
    """
    Calcula la media móvil de una lista de valores.
    """
    return np.convolve(lista, np.ones(ventana)/ventana, mode='valid')


# ------------------------------------------------------
# Gráfica de recompensas + media móvil
# ------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(recompensas, label="Recompensa por episodio", alpha=0.5)
plt.plot(media_movil(recompensas), label="Media móvil (20)", color='orange')
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("Desempeño del Agente (DDQN en Pong ALE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("DDQN_Atari.png")
plt.show()