# ==============================================
# NES aplicado al entorno ALE/Pong-v5
# ==============================================


# ===================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ===================================================================
# import gymnasium as gym  # Entorno de simulación ALE
import numpy as np  # Operaciones numéricas
import torch  # Tensores y cómputo con GPU
import torch.nn as nn  # Módulos para redes neuronales
import matplotlib.pyplot as plt  # Visualización
import cv2  # Procesamiento de imágenes
from collections import deque  # Cola de tamaño fijo
import gymnasium as gym  # Importa la librería Gymnasium para crear entornos de entrenamiento como Pong
import time  # Medición de tiempo (opcional)
import ale_py  # Backend de Atari usado por Gym

# ------------------------------------------------------
# Preprocesamiento de frames 
# ------------------------------------------------------
def preprocesar(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convertir a escala de grises
    redimensionada = cv2.resize(gris, (84, 84), interpolation=cv2.INTER_AREA)  # Redimensionar
    normalizada = redimensionada.astype(np.float32) / 255.0  # Normalizar entre 0 y 1
    return normalizada  # Retornar frame procesado

# ------------------------------------------------------
# Red convolucional para NES
# ------------------------------------------------------
class PoliticaNES(nn.Module):
    def __init__(self, num_acciones):
        super().__init__()  # Inicializa clase base
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # Capa conv1
        self.relu1 = nn.ReLU()  # Activación
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # Capa conv2
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # Capa conv3
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()  # Aplanado
        self.fc1 = nn.Linear(7 * 7 * 64, 512)  # Capa totalmente conectada
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(512, num_acciones)  # Capa de salida

    def forward(self, x):
        x = self.relu1(self.conv1(x))  # Paso por capa conv1
        x = self.relu2(self.conv2(x))  # Paso por capa conv2
        x = self.relu3(self.conv3(x))  # Paso por capa conv3
        x = self.flatten(x)  # Aplana para capa densa
        x = self.relu4(self.fc1(x))  # Capa oculta
        return self.out(x)  # Salida con logits por acción

    def seleccionar_accion(self, estado, dispositivo):
        estado_tensor = torch.tensor(np.array([estado]), dtype=torch.float32, device=dispositivo)  # Estado a tensor
        logits = self.forward(estado_tensor)  # Forward pass
        return torch.argmax(logits, dim=1).item()  # Acción con mayor valor

# ------------------------------------------------------
# Funciones utilitarias para NES
# ------------------------------------------------------
def obtener_parametros_flat(modelo):
    return torch.cat([p.data.view(-1) for p in modelo.parameters()])  # Concatenar todos los parámetros

def asignar_parametros_flat(modelo, vector):
    indice = 0  # Índice de recorrido
    for p in modelo.parameters():  # Por cada parámetro
        num = p.numel()  # Número de elementos
        p.data.copy_(vector[indice:indice+num].view(p.size()))  # Asignar porciones
        indice += num  # Avanzar índice

# ------------------------------------------------------
# Evaluar una política en un episodio
# ------------------------------------------------------
def evaluar_politica(env, politica, dispositivo, render=False):
    obs, _ = env.reset()  # Reiniciar entorno
    frame = preprocesar(obs)  # Preprocesar primer frame
    frames = deque([frame] * 4, maxlen=4)  # Inicializar pila de frames
    estado = np.stack(frames, axis=0)  # Formar estado inicial

    total_recompensa = 0  # Inicializar recompensa
    terminado = False  # Estado del episodio

    while not terminado:
        accion = politica.seleccionar_accion(estado, dispositivo)  # Elegir acción
        obs_, recompensa, termin, trunc, _ = env.step(accion)  # Ejecutar acción
        terminado = termin or trunc  # Verificar finalización
        frame_ = preprocesar(obs_)  # Preprocesar siguiente frame
        frames.append(frame_)  # Agregar a pila
        estado = np.stack(frames, axis=0)  # Nuevo estado
        total_recompensa += recompensa  # Acumular recompensa
        if render:
            env.render()  # Mostrar entorno si se desea

    return total_recompensa  # Retornar recompensa total

# ------------------------------------------------------
# Entrenamiento NES para Pong
# ------------------------------------------------------

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Seleccionar dispositivo
env = gym.make("ALE/Pong-v5")  # Crear entorno Pong
num_acciones = env.action_space.n  # Número de acciones posibles

politica = PoliticaNES(num_acciones).to(dispositivo)  # Inicializar red

EPISODIOS = 1000  # Total de episodios
PERTURBACIONES = 20  # Número de perturbaciones por episodio
SIGMA = 0.05  # Desviación del ruido
ALPHA = 0.01  # Tasa de aprendizaje

recompensas_por_episodio = []  # Lista para registrar recompensas

for ep in range(EPISODIOS):  # Loop principal
    theta = obtener_parametros_flat(politica)  # Obtener parámetros actuales
    perturbaciones = []  # Lista de ruidos
    recompensas = []  # Lista de recompensas

    for _ in range(PERTURBACIONES):  # Perturbaciones por episodio
        ruido = torch.randn_like(theta)  # Generar ruido
        theta_perturbado = theta + SIGMA * ruido  # Aplicar ruido
        asignar_parametros_flat(politica, theta_perturbado)  # Asignar nuevos parámetros
        R = evaluar_politica(env, politica, dispositivo)  # Evaluar política
        perturbaciones.append(ruido)  # Guardar ruido
        recompensas.append(R)  # Guardar recompensa

    recompensas_tensor = torch.tensor(recompensas, dtype=torch.float32)  # Convertir a tensor
    recompensas_normalizadas = (recompensas_tensor - recompensas_tensor.mean()) / (recompensas_tensor.std() + 1e-8)  # Normalizar

    gradiente = torch.zeros_like(theta)  # Inicializar gradiente
    for Rn, ruido in zip(recompensas_normalizadas, perturbaciones):  # Estimar gradiente
        gradiente += Rn * ruido
    gradiente /= (PERTURBACIONES * SIGMA)  # Escalar gradiente

    theta += ALPHA * gradiente  # Actualizar parámetros
    asignar_parametros_flat(politica, theta)  # Asignar a la red

    recompensa_promedio = np.mean(recompensas)  # Calcular recompensa promedio
    recompensas_por_episodio.append(recompensa_promedio)  # Registrar
    print(f"Episodio {ep+1}/{EPISODIOS} - Recompensa promedio: {recompensa_promedio:.2f}")  # Mostrar progreso

# ------------------------------------------------------
# Visualización de resultados
# ------------------------------------------------------
plt.figure(figsize=(10, 5))  # Tamaño del gráfico
plt.plot(recompensas_por_episodio, label="Recompensa por episodio")  # Línea base

ventana = 20  # Tamaño de la media móvil
if len(recompensas_por_episodio) >= ventana:
    media_movil = np.convolve(recompensas_por_episodio, np.ones(ventana)/ventana, mode='valid')  # Calcular media móvil
    plt.plot(range(ventana - 1, len(recompensas_por_episodio)), media_movil, color='orange', linewidth=2, label=f"Media móvil ({ventana})")  # Línea suavizada

plt.xlabel("Episodio")  # Etiqueta eje X
plt.ylabel("Recompensa Total")  # Etiqueta eje Y
plt.title("Desempeño del Agente (NES en Pong)")  # Título
plt.grid(True)  # Cuadrícula
plt.legend()  # Leyenda
plt.tight_layout()  # Ajuste
plt.savefig("NES_Atari.png")  # Guardar imagen
plt.show()  # Mostrar gráfico