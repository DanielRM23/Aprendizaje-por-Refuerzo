# 🔹 Importación de librerías
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from minatar import Environment

# ===================================================================
# 🔹 CLASE: Red Neuronal para Deep Q-Network (DQN)
# ===================================================================
class RedQ(nn.Module):  # La clase hereda de nn.Module (base de redes en PyTorch)
    def __init__(self, forma_entrada, num_acciones):
        """
        Inicializa la red neuronal para aproximar la función Q.

        Parámetros:
        - forma_entrada: Dimensión del estado del juego (ej. [10,10,4])
        - num_acciones: Cantidad de acciones posibles en el entorno
        """
        super(RedQ, self).__init__()

        # 🔹 Capa convolucional 1: Detecta patrones espaciales en la imagen
        self.capa_conv1 = nn.Conv2d(in_channels=forma_entrada[2], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.activacion1 = nn.ReLU()  # Introduce no linealidad

        # 🔹 Capa convolucional 2: Extrae características más profundas
        self.capa_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.activacion2 = nn.ReLU()

        # 🔹 Aplanado para convertir en vector
        self.aplanado = nn.Flatten()

        # 🔹 Primera capa densa: Reduce dimensionalidad
        self.capa_densa1 = nn.Linear(64 * forma_entrada[0] * forma_entrada[1], 256)
        self.activacion3 = nn.ReLU()

        # 🔹 Capa de salida: Predice valores Q para cada acción posible
        self.capa_salida = nn.Linear(256, num_acciones)

    # 🔹 Método para la propagación hacia adelante (cómo fluye la información en la red)
    def forward(self, x):
        # ⚠️ PyTorch espera entradas en formato (batch, canales, alto, ancho)
        # ⚠️ MinAtar devuelve (batch, alto, ancho, canales), por eso hacemos:
        x = x.permute(0, 3, 1, 2)  # Reorganiza los ejes: (batch, 10, 10, 4) → (batch, 4, 10, 10)
        
        x = self.activacion1(self.capa_conv1(x))  # Paso por la primera convolución y activación
        x = self.activacion2(self.capa_conv2(x))  # Paso por la segunda convolución y activación
        
        x = self.aplanado(x)  # Aplanar la salida antes de pasar a capas densas
        x = self.activacion3(self.capa_densa1(x))  # Paso por la primera capa densa
        
        return self.capa_salida(x)  # Devuelve los valores Q para cada acción

    #  Entrada: Estado del juego (10x10x4)
    #    ⬇️  (permute)
    # Reorganización: (4, 10, 10)
    #    ⬇️  (capa_conv1)
    # Convolución 1: 32 filtros → (32, 10, 10)
    #    ⬇️  (capa_conv2)
    # Convolución 2: 64 filtros → (64, 10, 10)
    #    ⬇️  (aplanado)
    # Aplanado: 6400 valores
    #    ⬇️  (capa_densa1)
    # Capa densa: 256 valores
    #    ⬇️  (capa_salida)
    # Capa de salida: 6 valores Q (uno por acción)

# ===================================================================
# 🔹 CLASE: Buffer de Experiencia (Replay Buffer)
# ===================================================================
class BufferExperiencia:
    """
    Buffer de experiencia para almacenar las interacciones del agente con el entorno.
    """

    def __init__(self, capacidad_maxima):
        self.buffer = deque(maxlen=capacidad_maxima)  # FIFO buffer con tamaño limitado

    def agregar(self, estado, accion, recompensa, nuevo_estado, terminado):
        """
        Agrega una experiencia al buffer.
        """
        self.buffer.append((estado, accion, recompensa, nuevo_estado, terminado))

    def obtener_muestra(self, tamaño_lote):
        """
        Devuelve una muestra aleatoria de experiencias del buffer.
        """
        return random.sample(self.buffer, tamaño_lote) if len(self.buffer) >= tamaño_lote else list(self.buffer)

    def __len__(self):
        """
        Retorna la cantidad de experiencias almacenadas en el buffer.
        """
        return len(self.buffer)

# ===================================================================
# 🔹 CONFIGURACIÓN DEL ENTORNO (MinAtar - Breakout)
# ===================================================================
entorno_juego = Environment("breakout")
num_acciones_posibles = entorno_juego.num_actions()
forma_estado_juego = entorno_juego.state_shape()
dimension_estado = np.prod(forma_estado_juego)

# 🔹 Imprimir información del entorno
print(f"Número de acciones posibles: {num_acciones_posibles}")
print(f"Forma del estado del juego: {forma_estado_juego}")

# 🔹 Obtener el estado inicial
estado_actual = entorno_juego.state()
print(f"Dimensión del estado inicial: {estado_actual.shape}")

# ===================================================================
# 🔹 CONFIGURACIÓN DEL MODELO Y BUFFER
# ===================================================================
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Crear la red principal y la red objetivo
red_principal = RedQ(forma_entrada=forma_estado_juego, num_acciones=num_acciones_posibles).to(dispositivo)
red_objetivo = RedQ(forma_entrada=forma_estado_juego, num_acciones=num_acciones_posibles).to(dispositivo)
red_objetivo.load_state_dict(red_principal.state_dict())  # Copiar pesos de la red principal
red_objetivo.eval()  # La red objetivo no se entrena directamente

# 🔹 Configurar el optimizador y la función de pérdida
optimizador = optim.Adam(red_principal.parameters(), lr=0.001)
funcion_perdida = nn.MSELoss()

# 🔹 Inicializar el buffer de experiencia
buffer_replay = BufferExperiencia(capacidad_maxima=100000)

print("Modelo y buffer de experiencia creados. Listos para entrenar.")

# ===================================================================
# 🔹 PRUEBA DEL BUFFER DE EXPERIENCIA
# ===================================================================
# 🔹 Simular 15 experiencias y agregarlas al buffer
for i in range(15):
    estado_falso = np.random.rand(10, 10, 4)  # Estado aleatorio de forma (10,10,4)
    accion_falsa = np.random.randint(6)  # Acción aleatoria entre 0 y 5
    recompensa_falsa = np.random.random()  # Recompensa aleatoria entre 0 y 1
    nuevo_estado_falso = np.random.rand(10, 10, 4)  # Nuevo estado aleatorio
    terminado_falso = random.choice([True, False])  # Aleatoriamente True o False

    buffer_replay.agregar(estado_falso, accion_falsa, recompensa_falsa, nuevo_estado_falso, terminado_falso)
    
    # 🔹 Imprimir información del buffer después de cada inserción
    print(f"Experiencia {i+1} agregada. Tamaño actual del buffer: {len(buffer_replay)}")

# 🔹 Obtener una muestra aleatoria de 5 experiencias
muestra = buffer_replay.obtener_muestra(5)

# 🔹 Mostrar el contenido de una experiencia de la muestra
print("\nEjemplo de experiencia extraída del buffer:")
estado_ejemplo, accion_ejemplo, recompensa_ejemplo, nuevo_estado_ejemplo, terminado_ejemplo = muestra[0]

print(f"Acción tomada: {accion_ejemplo}")
print(f"Recompensa recibida: {recompensa_ejemplo}")
print(f"Episodio terminado: {terminado_ejemplo}")
print(f"Forma del estado: {estado_ejemplo.shape}")


import numpy as np

class PoliticaEpsilonGreedy:
    """
    Implementa la estrategia ε-greedy para la selección de acciones en DDQN.
    """

    def __init__(self, num_acciones, epsilon_inicial=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Inicializa la política ε-greedy.

        Parámetros:
        - num_acciones: Número total de acciones posibles en el entorno.
        - epsilon_inicial: Valor inicial de ε (alta exploración).
        - epsilon_min: Valor mínimo de ε (cuando el agente ya ha aprendido lo suficiente).
        - epsilon_decay: Factor de decaimiento de ε (reduce la exploración gradualmente).
        """
        self.num_acciones = num_acciones
        self.epsilon = epsilon_inicial  # Exploración inicial alta
        self.epsilon_min = epsilon_min  # Exploración mínima al final del entrenamiento
        self.epsilon_decay = epsilon_decay  # Factor de reducción de ε en cada episodio

    def seleccionar_accion(self, estado, red_q, dispositivo):
        """
        Selecciona una acción usando la estrategia ε-greedy.

        Parámetros:
        - estado: Estado actual del entorno (formato (10,10,4)).
        - red_q: Red neuronal principal que estima valores Q.
        - dispositivo: GPU o CPU donde se ejecuta el modelo.

        Retorna:
        - acción seleccionada (int)
        """
        if np.random.rand() < self.epsilon:
            # 🔹 Exploración: Elegir acción aleatoria
            return np.random.randint(self.num_acciones)
        else:
            # 🔹 Explotación: Elegir la mejor acción según la red Q
            estado_tensor = torch.tensor(estado, dtype=torch.float32, device=dispositivo).unsqueeze(0)  
            with torch.no_grad():  # No necesitamos gradientes aquí
                valores_q = red_q(estado_tensor)
            return torch.argmax(valores_q).item()  # Retorna la acción con el mayor valor Q

    def actualizar_epsilon(self):
        """
        Reduce el valor de ε después de cada episodio.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  # Nunca baja de ε_min


# 🔹 Inicializar la política ε-greedy
politica = PoliticaEpsilonGreedy(num_acciones=num_acciones_posibles)

# 🔹 Obtener una acción con la estrategia ε-greedy
accion = politica.seleccionar_accion(estado_actual, red_principal, dispositivo)
print(f"Acción seleccionada por ε-greedy: {accion}")

# 🔹 Simular la actualización de ε después de 5 episodios
for episodio in range(5):
    politica.actualizar_epsilon()
    print(f"Episodio {episodio+1}: ε = {politica.epsilon}")



