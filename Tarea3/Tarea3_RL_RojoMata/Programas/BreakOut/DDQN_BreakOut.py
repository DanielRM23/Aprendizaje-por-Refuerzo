# ==============================================
# DDQN aplicado al entorno breakout
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

# ===================================================================
# CLASE: Red Neuronal para Deep Q-Network (DQN)
# ===================================================================
class RedQ(nn.Module):
    """
    Red neuronal convolucional que estima los valores Q para cada acción posible
    dado un estado del entorno.
    """
    def __init__(self, forma_entrada, num_acciones):
        """
        Inicializa la arquitectura de la red neuronal.

        Parámetros:
        - forma_entrada: Dimensión del estado de entrada (alto, ancho, canales).
        - num_acciones: Número total de acciones posibles.
        """
        super(RedQ, self).__init__()
        self.capa_conv1 = nn.Conv2d(in_channels=forma_entrada[2], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.activacion1 = nn.ReLU()
        self.capa_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.activacion2 = nn.ReLU()
        self.aplanado = nn.Flatten()
        self.capa_densa1 = nn.Linear(64 * forma_entrada[0] * forma_entrada[1], 256)
        self.activacion3 = nn.ReLU()
        self.capa_salida = nn.Linear(256, num_acciones)

    def forward(self, x):
        """
        Define el flujo de datos a través de la red neuronal.

        Parámetros:
        - x: Tensor de entrada con forma (batch, alto, ancho, canales).

        Retorna:
        - Tensor de salida con los valores Q estimados por acción.
        """
        x = x.permute(0, 3, 1, 2)  # Reorganiza a (batch, canales, alto, ancho)
        x = self.activacion1(self.capa_conv1(x))
        x = self.activacion2(self.capa_conv2(x))
        x = self.aplanado(x)
        x = self.activacion3(self.capa_densa1(x))
        return self.capa_salida(x)

# ===================================================================
# CLASE: Buffer de Experiencia (Replay Buffer)
# ===================================================================
class BufferExperiencia:
    """
    Buffer de experiencia para almacenar transiciones de entrenamiento del agente.
    Utiliza una estructura FIFO para mantener experiencias recientes.
    """
    def __init__(self, capacidad_maxima):
        """
        Inicializa el buffer de experiencia.

        Parámetros:
        - capacidad_maxima: Cantidad máxima de transiciones a almacenar.
        """
        self.buffer = deque(maxlen=capacidad_maxima)

    def agregar(self, estado, accion, recompensa, nuevo_estado, terminado):
        """
        Agrega una nueva experiencia al buffer.

        Parámetros:
        - estado: Estado actual.
        - accion: Acción tomada.
        - recompensa: Recompensa recibida.
        - nuevo_estado: Estado posterior a la acción.
        - terminado: Booleano que indica si el episodio terminó.
        """
        self.buffer.append((estado, accion, recompensa, nuevo_estado, terminado))

    def obtener_muestra(self, tamaño_lote):
        """
        Obtiene una muestra aleatoria del buffer.

        Parámetros:
        - tamaño_lote: Cantidad de experiencias a devolver.

        Retorna:
        - Lista de experiencias.
        """
        return random.sample(self.buffer, tamaño_lote) if len(self.buffer) >= tamaño_lote else list(self.buffer)

    def __len__(self):
        """
        Devuelve la cantidad actual de experiencias almacenadas.
        """
        return len(self.buffer)

# ===================================================================
# CLASE: Política ε-greedy
# ===================================================================
class PoliticaEpsilonGreedy:
    """
    Implementa la estrategia epsilon-greedy para exploración y explotación en el entorno.
    """
    def __init__(self, num_acciones, epsilon_inicial=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Inicializa la política epsilon-greedy.

        Parámetros:
        - num_acciones: Número de acciones posibles.
        - epsilon_inicial: Valor inicial de exploración.
        - epsilon_min: Límite inferior de epsilon.
        - epsilon_decay: Factor de decaimiento por episodio.
        """
        self.num_acciones = num_acciones
        self.epsilon = epsilon_inicial
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def seleccionar_accion(self, estado, red_q, dispositivo):
        """
        Selecciona una acción basada en la política ε-greedy.

        Parámetros:
        - estado: Estado actual del entorno.
        - red_q: Red neuronal principal que estima valores Q.
        - dispositivo: CPU o GPU donde se ejecuta la red.

        Retorna:
        - Acción seleccionada (int).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_acciones)
        else:
            estado_tensor = torch.tensor(estado, dtype=torch.float32, device=dispositivo).unsqueeze(0)
            with torch.no_grad():
                valores_q = red_q(estado_tensor)
            return torch.argmax(valores_q).item()

    def actualizar_epsilon(self):
        """
        Aplica decaimiento exponencial a epsilon después de cada episodio.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# ===================================================================
# FUNCIÓN: Actualización de valores Q usando DDQN
# ===================================================================
def actualizar_q_values(red_principal, red_objetivo, buffer_replay, optimizador, funcion_perdida, dispositivo, gamma=0.99, batch_size=32):
    """
    Actualiza la red Q principal utilizando el algoritmo DDQN.

    Parámetros:
    - red_principal: Red Q que se entrena.
    - red_objetivo: Red Q objetivo (target network).
    - buffer_replay: Buffer de experiencia.
    - optimizador: Optimizador para actualizar los pesos de la red principal.
    - funcion_perdida: Función de pérdida a minimizar (e.g., MSE).
    - dispositivo: CPU o GPU.
    - gamma: Factor de descuento.
    - batch_size: Tamaño del lote de entrenamiento.
    """
    if len(buffer_replay) < batch_size:
        return  # No se actualiza si no hay suficientes datos

    batch = [exp for exp in buffer_replay.obtener_muestra(batch_size) if exp[0] is not None and exp[3] is not None]
    if len(batch) < batch_size:
        return

    estados, acciones, recompensas, nuevos_estados, terminados = zip(*batch)

    estados = torch.tensor(np.stack(estados), dtype=torch.float32, device=dispositivo)
    nuevos_estados = torch.tensor(np.stack(nuevos_estados), dtype=torch.float32, device=dispositivo)
    acciones = torch.tensor(acciones, dtype=torch.int64, device=dispositivo).unsqueeze(1)
    recompensas = torch.tensor(recompensas, dtype=torch.float32, device=dispositivo).unsqueeze(1)
    terminados = torch.tensor(terminados, dtype=torch.float32, device=dispositivo).unsqueeze(1)

    valores_q_actuales = red_principal(estados).gather(1, acciones)
    with torch.no_grad():
        mejores_acciones = red_principal(nuevos_estados).argmax(dim=1, keepdim=True)
        valores_q_futuros = red_objetivo(nuevos_estados).gather(1, mejores_acciones)
        q_objetivo = recompensas + gamma * valores_q_futuros * (1 - terminados)

    perdida = funcion_perdida(valores_q_actuales, q_objetivo)
    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

# ===================================================================
# ENTRENAMIENTO DEL AGENTE DDQN
# ===================================================================

# Inicializar el entorno MinAtar Breakout
entorno_juego = Environment("breakout")
num_acciones_posibles = entorno_juego.num_actions()  # Número de acciones que el agente puede tomar
forma_estado_juego = entorno_juego.state_shape()  # Dimensiones del estado observado

# Crear redes Q (principal y objetivo)
red_principal = RedQ(forma_entrada=forma_estado_juego, num_acciones=num_acciones_posibles)
red_objetivo = RedQ(forma_entrada=forma_estado_juego, num_acciones=num_acciones_posibles)
red_objetivo.load_state_dict(red_principal.state_dict())  # Copiar pesos de la red principal a la red objetivo
red_objetivo.eval()  # La red objetivo no se entrena directamente

# Configurar dispositivo para GPU si está disponible
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
red_principal.to(dispositivo)
red_objetivo.to(dispositivo)

# Inicializar buffer de experiencia y política epsilon-greedy
buffer_replay = BufferExperiencia(capacidad_maxima=100000)
politica = PoliticaEpsilonGreedy(num_acciones=num_acciones_posibles)

# Configurar optimizador y función de pérdida
optimizador = optim.Adam(red_principal.parameters(), lr=0.001)
funcion_perdida = nn.MSELoss()

# Hiperparámetros
num_episodios = 1000  # Número total de episodios de entrenamiento
gamma = 0.99  # Factor de descuento
batch_size = 32  # Tamaño del lote
actualizar_red_objetivo_cada = 1000  # Frecuencia para actualizar red objetivo (en pasos)
pasos_totales = 0  # Contador de pasos de entrenamiento
recompensas_totales = []  # Historial de recompensas

# Bucle principal de entrenamiento
for episodio in range(1, num_episodios + 1):
    entorno_juego.reset()  # Reiniciar el entorno
    estado_actual = entorno_juego.state()  # Obtener el primer estado

    # Verificar si el estado es válido; reintentar si es None
    intentos_reset = 0
    while estado_actual is None and intentos_reset < 10:
        entorno_juego.reset()
        estado_actual = entorno_juego.state()
        intentos_reset += 1
    if estado_actual is None:
        continue  # Omitir episodio si no se obtiene estado válido

    recompensa_total = 0  # Recompensa acumulada del episodio
    terminado = False  # Bandera de finalización

    # Bucle por pasos dentro del episodio
    while not terminado:
        accion = politica.seleccionar_accion(estado_actual, red_principal, dispositivo)  # Elegir acción
        recompensa, terminado = entorno_juego.act(accion)  # Ejecutar acción
        nuevo_estado = entorno_juego.state()  # Obtener nuevo estado

        # Agregar experiencia al buffer si es válida
        if estado_actual is not None and nuevo_estado is not None:
            buffer_replay.agregar(estado_actual, accion, recompensa, nuevo_estado, terminado)

        # Actualizar red Q principal usando DDQN
        actualizar_q_values(red_principal, red_objetivo, buffer_replay, optimizador, funcion_perdida, dispositivo, gamma, batch_size)

        estado_actual = nuevo_estado  # Mover al siguiente estado
        recompensa_total += recompensa  # Sumar recompensa
        pasos_totales += 1

        # Actualizar red objetivo periódicamente
        if pasos_totales % actualizar_red_objetivo_cada == 0:
            red_objetivo.load_state_dict(red_principal.state_dict())

    politica.actualizar_epsilon()  # Decaer epsilon para menos exploración
    recompensas_totales.append(recompensa_total)  # Registrar recompensa del episodio

    # Mostrar progreso en consola
    print(f"Episodio {episodio}/{num_episodios} - Recompensa total: {recompensa_total:.2f} - ε: {politica.epsilon:.4f}")
    time.sleep(0.01)  # Esperar brevemente para evitar sobrecarga

print("Entrenamiento completado 🎉")

# ===================================================================
# GRAFICAR RESULTADOS
# ===================================================================
ventana = 20  # Tamaño de ventana para media móvil
promedios_moviles = np.convolve(recompensas_totales, np.ones(ventana)/ventana, mode='valid')  # Suavizado

# Crear gráfica
plt.plot(recompensas_totales, label='Recompensa por episodio')
plt.plot(promedios_moviles, label=f'Media móvil ({ventana})')
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Desempeño del Agente (DDQN en MinAtar Breakout)")
plt.legend()
plt.savefig("DDQN_BrakOut.py")
plt.show()
