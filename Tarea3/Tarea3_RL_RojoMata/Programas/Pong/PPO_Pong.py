# ==============================================
# PPO aplicado al entorno ALE/Pong-v5
# ==============================================



# ===================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ===================================================================
import gymnasium as gym  # Importa la librería Gymnasium para crear entornos de entrenamiento como Pong
import numpy as np  # Importa NumPy para operaciones numéricas
import torch  # Importa PyTorch para operaciones con tensores y deep learning
import torch.nn as nn  # Importa módulos de redes neuronales de PyTorch
import torch.optim as optim  # Importa optimizadores como Adam
import matplotlib.pyplot as plt  # Importa Matplotlib para graficar resultados
import cv2  # Importa OpenCV para procesar imágenes (por ejemplo, redimensionar y escalar frames)
from collections import deque  # Importa deque para mantener una secuencia de los últimos 4 frames
import ale_py  # Interfaz de Gym para ambientes Atari

# ============================
# Preprocesamiento de frames
# ============================
def preprocesar(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convierte la imagen RGB a escala de grises
    redimensionada = cv2.resize(gris, (84, 84), interpolation=cv2.INTER_AREA)  # Redimensiona el frame a 84x84
    normalizada = redimensionada.astype(np.float32) / 255.0  # Normaliza los valores de píxel entre 0 y 1
    return normalizada  # Devuelve el frame procesado

# ============================
# Red Actor-Crítico
# ============================
class RedActor(nn.Module):  # Red neuronal que representa la política del agente
    def __init__(self, num_acciones):
        super().__init__()
        # Capas convolucionales para extraer características espaciales
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),  # Capa conv1: 4 canales de entrada, 32 filtros
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),  # Capa conv2
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()   # Capa conv3
        )
        # Capas completamente conectadas
        self.fc = nn.Sequential(
            nn.Flatten(),  # Aplana el tensor de salida convolucional
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),  # Capa oculta densa
            nn.Linear(512, num_acciones),  # Capa de salida: logits para cada acción
            nn.Softmax(dim=-1)  # Aplica softmax para obtener probabilidades
        )

    def forward(self, x):
        x = self.conv(x)  # Pasa por capas convolucionales
        return self.fc(x)  # Pasa por capas densas y devuelve distribución de acción

class RedCritica(nn.Module):  # Red neuronal para estimar el valor del estado (función V)
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, 1)  # Salida: valor escalar del estado
        )

    def forward(self, x):
        x = self.conv(x)  # Extrae características con convolucionales
        return self.fc(x)  # Calcula el valor del estado

# ============================
# Memoria PPO
# ============================
class MemoriaPPO:  # Estructura para almacenar las transiciones del agente
    def __init__(self):
        self.estados = []  # Lista de estados
        self.acciones = []  # Lista de acciones tomadas
        self.log_probs = []  # Log-probs de las acciones tomadas
        self.recompensas = []  # Recompensas obtenidas
        self.dones = []  # Indicadores de si el episodio terminó

    def guardar(self, estado, accion, log_prob, recompensa, terminado):
        self.estados.append(estado)  # Guarda estado
        self.acciones.append(accion)  # Guarda acción
        self.log_probs.append(log_prob.detach())  # Guarda log_prob desconectado del grafo
        self.recompensas.append(recompensa)  # Guarda recompensa
        self.dones.append(terminado)  # Guarda indicador de finalización

    def limpiar(self):
        self.__init__()  # Limpia todas las listas reinicializando el objeto

# ============================
# Utilidades PPO
# ============================
def calcular_ventajas(recompensas, valores, dones, gamma=0.99, lam=0.95):
    ventajas = np.zeros(len(recompensas))  # Inicializa arreglo de ventajas
    gae = 0  # Acumulador para Generalized Advantage Estimation (GAE)
    for t in reversed(range(len(recompensas))):  # Recorre de atrás hacia adelante
        delta = recompensas[t] + gamma * valores[t+1] * (1 - dones[t]) - valores[t]  # TD error
        gae = delta + gamma * lam * (1 - dones[t]) * gae  # GAE recursivo
        ventajas[t] = gae  # Guarda ventaja en t
    return ventajas  # Devuelve arreglo de ventajas

def funcion_perdida_ppo(logs_ant, logs_nuevos, ventajas, epsilon=0.2):
    ratio = torch.exp(logs_nuevos - logs_ant)  # Calcula ratio r_theta
    clip_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)  # Aplica clipping para estabilidad
    return -torch.min(ratio * ventajas, clip_ratio * ventajas).mean()  # PPO clipped objective

def crear_minibatches(total, tamano):
    indices = np.arange(total)  # Crea arreglo con índices
    np.random.shuffle(indices)  # Mezcla aleatoriamente
    for i in range(0, total, tamano):  # Crea lotes del tamaño indicado
        yield indices[i:i + tamano]  # Devuelve lote actual

# ============================
# Configuración
# ============================
env = gym.make("ALE/Pong-v5", render_mode=None)  # Crea el entorno Pong de Atari con Gym

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU si está disponible
num_acciones = env.action_space.n  # Obtiene el número de acciones posibles

actor = RedActor(num_acciones).to(dispositivo)  # Inicializa y envía la red actor al dispositivo
critico = RedCritica().to(dispositivo)  # Inicializa y envía la red crítica al dispositivo
opt_actor = optim.Adam(actor.parameters(), lr=1e-4)  # Optimizador para la red actor
opt_critico = optim.Adam(critico.parameters(), lr=1e-4)  # Optimizador para la red crítica
memoria = MemoriaPPO()  # Crea la memoria PPO

EPISODIOS = 500  # Número total de episodios
PASOS_BATCH = 2048  # Número de pasos entre cada actualización
K_epocas = 4  # Épocas por actualización PPO
MINIBATCH = 64  # Tamaño del mini-batch
beta_entropia = 0.01  # Peso para la entropía (exploración)

recompensas_totales = []  # Lista para registrar recompensas por episodio
pasos = 0  # Contador de pasos
episodio = 0  # Contador de episodios

# ============================
# Entrenamiento
# ============================
while pasos < 1_000_000:  # Repetir hasta alcanzar 1 millón de pasos
    obs, _ = env.reset()  # Reinicia el entorno y obtiene primer frame
    frame = preprocesar(obs)  # Aplica preprocesamiento
    frames = deque([frame] * 4, maxlen=4)  # Crea secuencia de 4 frames iguales
    estado = np.stack(frames, axis=0)  # Stack para crear tensor 4x84x84

    recompensa_ep = 0  # Recompensa acumulada del episodio
    terminado = False  # Bandera de finalización del episodio

    while not terminado:  # Repite hasta que el episodio termine
        estado_tensor = torch.tensor(np.array([estado]), dtype=torch.float32).to(dispositivo)  # Estado a tensor
        probs = actor(estado_tensor)  # Calcula distribución de acción
        dist = torch.distributions.Categorical(probs)  # Distribución categórica
        accion = dist.sample()  # Muestra una acción
        log_prob = dist.log_prob(accion)  # Log probabilidad de la acción

        obs_, recompensa, termin, trunc, _ = env.step(accion.item())  # Ejecuta acción en el entorno
        terminado = termin or trunc  # Verifica si terminó el episodio
        frame_ = preprocesar(obs_)  # Preprocesa siguiente frame
        frames.append(frame_)  # Agrega nuevo frame
        estado_siguiente = np.stack(frames, axis=0)  # Actualiza estado

        memoria.guardar(estado_tensor, accion, log_prob, recompensa, terminado)  # Guarda transición
        recompensa_ep += recompensa  # Acumula recompensa
        pasos += 1  # Incrementa pasos totales
        estado = estado_siguiente  # Avanza al siguiente estado

        if pasos % PASOS_BATCH == 0:  # Si es momento de actualizar
            batch_estados = torch.cat(memoria.estados).to(dispositivo)  # Junta todos los estados
            batch_acciones = torch.tensor(memoria.acciones, dtype=torch.long).to(dispositivo)  # Acciones
            batch_log_probs = torch.cat(memoria.log_probs).to(dispositivo)  # Log probs

            with torch.no_grad():
                valores = critico(batch_estados).squeeze().cpu().numpy()  # Valores del crítico
            valores = np.append(valores, 0)  # Agrega V(s') = 0 al final

            ventajas = torch.tensor(calcular_ventajas(memoria.recompensas, valores, memoria.dones), dtype=torch.float32).to(dispositivo)  # Ventajas GAE
            ventajas = (ventajas - ventajas.mean()) / (ventajas.std() + 1e-8)  # Normaliza

            retornos = []  # Inicializa lista de retornos
            descuento = 0  # Valor de retorno acumulado
            for r, d in zip(reversed(memoria.recompensas), reversed(memoria.dones)):
                descuento = r + 0.99 * descuento * (1 - d)  # Retorno con descuento
                retornos.insert(0, descuento)  # Inserta al inicio
            retornos = torch.tensor(retornos, dtype=torch.float32).to(dispositivo)  # Tensor de retornos

            total = len(memoria.estados)
            for _ in range(K_epocas):  # K pasadas sobre los datos
                for idx in crear_minibatches(total, MINIBATCH):  # Mini-batches aleatorios
                    mini_est = batch_estados[idx]
                    mini_acc = batch_acciones[idx]
                    mini_log_ant = batch_log_probs[idx]
                    mini_vent = ventajas[idx]
                    mini_ret = retornos[idx]

                    probs = actor(mini_est)
                    dist = torch.distributions.Categorical(probs)
                    log_nuevos = dist.log_prob(mini_acc)
                    entropia = dist.entropy().mean()

                    loss_actor = funcion_perdida_ppo(mini_log_ant, log_nuevos, mini_vent)
                    loss_actor_total = loss_actor - beta_entropia * entropia  # Pérdida con entropía

                    valores_est = critico(mini_est).squeeze()
                    loss_critico = ((mini_ret - valores_est) ** 2).mean()  # MSE del crítico

                    opt_actor.zero_grad()
                    loss_actor_total.backward()
                    opt_actor.step()

                    opt_critico.zero_grad()
                    loss_critico.backward()
                    opt_critico.step()

            memoria.limpiar()  # Limpia memoria después de actualizar

    recompensa_ep = float(recompensa_ep)  # Conversión por seguridad
    recompensas_totales.append(recompensa_ep)  # Guarda recompensa total del episodio
    episodio += 1  # Incrementa contador de episodios
    if episodio % 10 == 0:
        print(f"Episodio {episodio} - Recompensa: {recompensa_ep}")  # Muestra progreso

# ============================
# Visualización con media móvil
# ============================
plt.figure(figsize=(10, 5))  # Tamaño del gráfico
plt.plot(recompensas_totales, label="Recompensa por episodio", alpha=0.6)  # Línea azul

ventana = 20
if len(recompensas_totales) >= ventana:
    media_movil = np.convolve(recompensas_totales, np.ones(ventana)/ventana, mode='valid')  # Suavizado
    plt.plot(range(ventana - 1, len(recompensas_totales)), media_movil, color='orange', linewidth=2, label=f"Media móvil ({ventana})")  # Línea naranja

plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Desempeño del Agente (PPO en ALE/Pong-v5)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("PPO_Atari.png")  # Guarda imagen
plt.show()  # Muestra gráfico
