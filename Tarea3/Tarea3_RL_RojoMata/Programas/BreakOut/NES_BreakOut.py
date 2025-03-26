# ==============================================
# NES aplicado al entorno breakout
# ==============================================


# ===================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ===================================================================import numpy as np  # Librería para operaciones numéricas
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
# CONFIGURACIÓN DEL ENTORNO
# ===================================================================

# Crear una instancia del entorno Breakout de MinAtar
entorno = Environment("breakout")

# Obtener la cantidad de acciones posibles que el agente puede tomar
num_acciones = entorno.num_actions()

# Obtener la forma (dimensiones) del estado que entrega el entorno
dim_forma_estado = entorno.state_shape()

# Calcular la dimensión total del estado aplanado
tamano_estado_flat = np.prod(dim_forma_estado)

# Imprimir información relevante del entorno
print(f"Número de acciones posibles: {num_acciones}")
print(f"Forma del estado del juego: {dim_forma_estado}")
print(f"Dimensión del estado inicial: {entorno.state().shape}")

# ===================================================================
# DEFINICIÓN DE LA RED DE POLÍTICA
# ===================================================================

class PoliticaNES(nn.Module):
    """
    Red neuronal simple que actúa como política para el agente NES.

    Parameters
    ----------
    dimension_entrada : int
        Tamaño del vector de entrada (estado aplanado).
    num_acciones_salida : int
        Número de acciones posibles en el entorno.

    Métodos
    -------
    forward(x)
        Propaga la entrada por la red y devuelve los logits por acción.
    seleccionar_accion(estado)
        Dado un estado, devuelve la acción elegida (con mayor valor).
    """
    def __init__(self, dimension_entrada, num_acciones_salida):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Linear(dimension_entrada, 64),  # Capa oculta de 64 neuronas
            nn.ReLU(),                         # Activación ReLU
            nn.Linear(64, num_acciones_salida) # Capa de salida
        )

    def forward(self, x):
        # Propagación hacia adelante del modelo
        return self.modelo(x)

    def seleccionar_accion(self, estado):
        # Convierte el estado en tensor y lo aplana
        estado_tensor = torch.tensor(estado.flatten(), dtype=torch.float32).unsqueeze(0)
        # Calcula los logits (valores) para cada acción
        logits = self.forward(estado_tensor)
        # Devuelve la acción con mayor valor
        accion = torch.argmax(logits, dim=1).item()
        return accion

# ===================================================================
# FUNCIONES DE UTILIDAD PARA NES
# ===================================================================

def obtener_parametros_flat(modelo):
    """
    Convierte todos los parámetros de una red neuronal en un solo vector plano (1D).

    Parameters
    ----------
    modelo : torch.nn.Module
        Modelo de red neuronal.

    Returns
    -------
    torch.Tensor
        Vector 1D con todos los parámetros concatenados.
    """
    return torch.cat([p.data.view(-1) for p in modelo.parameters()])


def asignar_parametros_flat(modelo, vector_parametros):
    """
    Asigna los valores de un vector plano a los parámetros de un modelo.

    Parameters
    ----------
    modelo : torch.nn.Module
        Modelo de red neuronal al que se le asignarán los parámetros.
    vector_parametros : torch.Tensor
        Vector plano con los valores de los parámetros a asignar.
    """
    indice = 0
    for p in modelo.parameters():
        cantidad_elementos = p.numel()
        porcion = vector_parametros[indice:indice + cantidad_elementos]
        p.data.copy_(porcion.view(p.size()))
        indice += cantidad_elementos


def evaluar_politica(env, politica, render=False):
    """
    Ejecuta un episodio completo usando una política dada y devuelve la recompensa total.

    Parameters
    ----------
    env : Environment
        Entorno de MinAtar.
    politica : PoliticaNES
        Modelo de política que se evaluará.
    render : bool, optional
        Si True, muestra el entorno en cada paso.

    Returns
    -------
    int
        Recompensa total acumulada durante el episodio.
    """
    env.reset()
    recompensa_total = 0

    # Acción inicial forzada: disparar (acción 5 en MinAtar Breakout)
    recompensa, terminado = env.act(5)
    recompensa_total += recompensa
    estado = env.state()

    # Ejecutar el episodio hasta que termine
    while not terminado:
        accion = politica.seleccionar_accion(estado)
        recompensa, terminado = env.act(accion)
        recompensa_total += recompensa
        estado = env.state()

        if render:
            print(env.display())

    return recompensa_total

# ===================================================================
# ENTRENAMIENTO CON NES
# ===================================================================

# Hiperparámetros del algoritmo
episodios_entrenamiento = 1000     # Número de iteraciones de entrenamiento
num_perturbaciones = 100           # Número de políticas perturbadas por episodio
desviacion_ruido = 0.1             # Magnitud del ruido (sigma)
tasa_aprendizaje = 0.01            # Velocidad de actualización (alpha)

# Crear la política inicial
politica = PoliticaNES(tamano_estado_flat, num_acciones)

# Lista para almacenar recompensas promedio por episodio
recompensas_promedio_por_episodio = []

# Bucle principal de entrenamiento
for ep in range(episodios_entrenamiento):
    # Obtener parámetros actuales de la red como vector plano
    theta_actual = obtener_parametros_flat(politica)

    lista_perturbaciones = []
    lista_recompensas = []

    # Generar y evaluar múltiples perturbaciones
    for _ in range(num_perturbaciones):
        ruido = torch.randn_like(theta_actual)  # Ruido gaussiano
        theta_alterado = theta_actual + desviacion_ruido * ruido  # Perturbación
        asignar_parametros_flat(politica, theta_alterado)         # Aplicar a la red

        recompensa = evaluar_politica(entorno, politica)          # Evaluar la política
        lista_perturbaciones.append(ruido)
        lista_recompensas.append(recompensa)

    # Calcular recompensa promedio cruda (sin normalizar)
    recompensa_promedio_cruda = sum(lista_recompensas) / num_perturbaciones
    recompensas_promedio_por_episodio.append(recompensa_promedio_cruda)

    # Normalizar recompensas para estabilidad numérica
    recompensas_tensor = torch.tensor(lista_recompensas, dtype=torch.float32)
    recompensas_normalizadas = (recompensas_tensor - recompensas_tensor.mean()) / (recompensas_tensor.std() + 1e-8)

    # Estimar gradiente como suma ponderada de las perturbaciones
    gradiente_estimado = torch.zeros_like(theta_actual)
    for R, eps in zip(recompensas_normalizadas, lista_perturbaciones):
        gradiente_estimado += R * eps
    gradiente_estimado /= (num_perturbaciones * desviacion_ruido)

    # Aplicar el gradiente estimado a los parámetros
    theta_actual += tasa_aprendizaje * gradiente_estimado
    asignar_parametros_flat(politica, theta_actual)

    # Mostrar progreso
    print(f"Episodio {ep+1}/{episodios_entrenamiento} - Recompensa promedio: {recompensa_promedio_cruda:.2f}")

# ===================================================================
# VISUALIZACIÓN DE RESULTADOS
# ===================================================================

# Graficar la recompensa promedio por episodio
plt.plot(recompensas_promedio_por_episodio)
plt.xlabel("Episodio")
plt.ylabel("Recompensa promedio")
plt.title("Rendimiento de NES en Breakout")
plt.grid()
plt.savefig("NES_BreakOut.png")
plt.show()
