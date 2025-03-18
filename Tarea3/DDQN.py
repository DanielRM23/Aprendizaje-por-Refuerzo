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



import torch


def actualizar_q_values(red_principal, red_objetivo, buffer_replay, optimizador, funcion_perdida, dispositivo, gamma=0.99, batch_size=32):
    """
    Actualiza la red principal utilizando DDQN.
    """
    if len(buffer_replay) < batch_size:
        return  # Esperar más experiencias

    batch = buffer_replay.obtener_muestra(batch_size)

    # 🔹 Filtrar estados None antes de procesarlos
    batch = [exp for exp in batch if exp[0] is not None and exp[3] is not None]

    # Si después de filtrar hay menos datos de los necesarios, no actualizar
    if len(batch) < batch_size:
        return  

    estados, acciones, recompensas, nuevos_estados, terminados = zip(*batch)

    # 🔹 Revisar que no haya `None` en los estados antes de transformarlos en tensores
    assert all(s is not None for s in estados), "Hay un estado None en el batch"
    assert all(s is not None for s in nuevos_estados), "Hay un nuevo_estado None en el batch"

    estados = torch.tensor(np.stack(estados, axis=0), dtype=torch.float32, device=dispositivo)
    nuevos_estados = torch.tensor(np.stack(nuevos_estados, axis=0), dtype=torch.float32, device=dispositivo)

    acciones = torch.tensor(acciones, dtype=torch.int64, device=dispositivo).unsqueeze(1)
    recompensas = torch.tensor(recompensas, dtype=torch.float32, device=dispositivo).unsqueeze(1)
    terminados = torch.tensor(terminados, dtype=torch.float32, device=dispositivo).unsqueeze(1)

    # 🔹 Calcular los valores Q actuales
    valores_q_actuales = red_principal(estados).gather(1, acciones)

    with torch.no_grad():
        mejores_acciones = red_principal(nuevos_estados).argmax(dim=1, keepdim=True)
        valores_q_futuros = red_objetivo(nuevos_estados).gather(1, mejores_acciones)
        q_objetivo = recompensas + gamma * valores_q_futuros * (1 - terminados)

    # 🔹 Calcular pérdida y actualizar pesos
    perdida = funcion_perdida(valores_q_actuales, q_objetivo)
    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()




import time

# 🔹 Parámetros de entrenamiento
num_episodios = 500  # Número total de episodios de entrenamiento
gamma = 0.99  # Factor de descuento para recompensas futuras
batch_size = 32  # Tamaño del lote para entrenar la red
actualizar_red_objetivo_cada = 1000  # Cada cuántos pasos copiamos `red_principal` → `red_objetivo`

# 🔹 Inicializar la política ε-greedy
politica = PoliticaEpsilonGreedy(num_acciones=num_acciones_posibles)

# 🔹 Contador de pasos totales
pasos_totales = 0


import time

# 🔹 Parámetros de entrenamiento
num_episodios = 500  # Número total de episodios de entrenamiento
gamma = 0.99  # Factor de descuento para recompensas futuras
batch_size = 32  # Tamaño del lote para entrenar la red
actualizar_red_objetivo_cada = 1000  # Cada cuántos pasos copiamos `red_principal` → `red_objetivo`

# 🔹 Inicializar la política ε-greedy
politica = PoliticaEpsilonGreedy(num_acciones=num_acciones_posibles)

# 🔹 Contador de pasos totales
pasos_totales = 0

# 🔹 Ciclo de entrenamiento
for episodio in range(1, num_episodios + 1):
    # 🔹 Reiniciar el entorno correctamente
    entorno_juego.reset()  # Resetear el entorno sin asignarlo a una variable
    estado_actual = entorno_juego.state()  # Obtener el estado real después del reset

    # 🔹 Verificar si `estado_actual` es None y reintentar si es necesario
    intentos_reset = 0
    while estado_actual is None and intentos_reset < 10:  # Limitar intentos para evitar bucles infinitos
        print(f"⚠️ Advertencia: `state()` devolvió None después de reset(). Intento {intentos_reset+1}/10...")
        entorno_juego.reset()
        estado_actual = entorno_juego.state()
        intentos_reset += 1

    if estado_actual is None:
        print("❌ Error crítico: No se pudo obtener un estado válido después de 10 intentos. Abortando episodio.")
        continue  # Saltar al siguiente episodio

    recompensa_total = 0  # Acumular la recompensa total del episodio
    terminado = False

    while not terminado:
        # 🔹 Asegurar que `estado_actual` no sea None antes de seleccionar la acción
        if estado_actual is None:
            print("⚠️ Advertencia: `estado_actual` es None dentro del episodio. Saliendo del loop...")
            break  # Salir del episodio si el estado es inválido

        # 🔹 Seleccionar acción con ε-greedy
        accion = politica.seleccionar_accion(estado_actual, red_principal, dispositivo)

        # 🔹 Ejecutar la acción en el entorno
        recompensa, terminado = entorno_juego.act(accion)
        nuevo_estado = entorno_juego.state()

        # 🔹 Filtrar estados `None` antes de agregar al buffer
        if estado_actual is not None and nuevo_estado is not None:
            buffer_replay.agregar(estado_actual, accion, recompensa, nuevo_estado, terminado)

        # 🔹 Actualizar la red neuronal si hay suficientes datos
        actualizar_q_values(red_principal, red_objetivo, buffer_replay, optimizador, funcion_perdida, dispositivo, gamma, batch_size)

        # 🔹 Mover al nuevo estado
        estado_actual = nuevo_estado
        recompensa_total += recompensa
        pasos_totales += 1

        # 🔹 Actualizar la red objetivo cada `actualizar_red_objetivo_cada` pasos
        if pasos_totales % actualizar_red_objetivo_cada == 0:
            red_objetivo.load_state_dict(red_principal.state_dict())

    # 🔹 Reducir ε después de cada episodio
    politica.actualizar_epsilon()

    # 🔹 Imprimir estadísticas del episodio
    print(f"Episodio {episodio}/{num_episodios} - Recompensa total: {recompensa_total:.2f} - ε: {politica.epsilon:.4f}")

    # 🔹 Pequeña pausa para evitar sobrecargar la CPU/GPU
    time.sleep(0.01)

print("Entrenamiento completado 🎉")


import matplotlib.pyplot as plt

# 🔹 Lista para almacenar las recompensas de cada episodio
recompensas_totales = []

# 🔹 Ciclo de entrenamiento
for episodio in range(1, num_episodios + 1):
    entorno_juego.reset()
    estado_actual = entorno_juego.state()
    
    intentos_reset = 0
    while estado_actual is None and intentos_reset < 10:
        estado_actual = entorno_juego.reset()
        estado_actual = entorno_juego.state()
        intentos_reset += 1

    if estado_actual is None:
        continue

    recompensa_total = 0
    terminado = False

    while not terminado:
        accion = politica.seleccionar_accion(estado_actual, red_principal, dispositivo)
        recompensa, terminado = entorno_juego.act(accion)
        nuevo_estado = entorno_juego.state()

        if estado_actual is not None and nuevo_estado is not None:
            buffer_replay.agregar(estado_actual, accion, recompensa, nuevo_estado, terminado)

        actualizar_q_values(red_principal, red_objetivo, buffer_replay, optimizador, funcion_perdida, dispositivo, gamma, batch_size)

        estado_actual = nuevo_estado
        recompensa_total += recompensa
        pasos_totales += 1

        if pasos_totales % actualizar_red_objetivo_cada == 0:
            red_objetivo.load_state_dict(red_principal.state_dict())

    politica.actualizar_epsilon()
    recompensas_totales.append(recompensa_total)

    print(f"Episodio {episodio}/{num_episodios} - Recompensa total: {recompensa_total:.2f} - ε: {politica.epsilon:.4f}")

    time.sleep(0.01)

# 🔹 Graficar la recompensa total por episodio
plt.plot(recompensas_totales)
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Desempeño del Agente (Recompensa por Episodio)")
plt.show()

