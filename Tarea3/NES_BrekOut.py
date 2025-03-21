# 🔹 Importación de librerías necesarias
import numpy as np
import random
import torch
import torch.nn as nn
from minatar import Environment
import matplotlib.pyplot as plt

# ===================================================================
# 🔹 CONFIGURACIÓN DEL ENTORNO (MinAtar - Breakout)
# ===================================================================

# Crear el entorno de Breakout usando MinAtar
entorno = Environment("breakout")

# Obtener información clave del entorno
num_acciones = entorno.num_actions()
dim_forma_estado = entorno.state_shape()
tamano_estado_flat = np.prod(dim_forma_estado)

# Mostrar información del entorno
print(f"Número de acciones posibles: {num_acciones}")
print(f"Forma del estado del juego: {dim_forma_estado}")
print(f"Dimensión del estado inicial: {entorno.state().shape}")

# ===================================================================
# 🔹 DEFINICIÓN DE LA RED DE POLÍTICA
# ===================================================================

class PoliticaNES(nn.Module):
    """
    Red neuronal simple que actúa como política.
    Entrada: estado aplanado
    Salida: logits para cada acción posible
    """
    def __init__(self, dimension_entrada, num_acciones_salida):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Linear(dimension_entrada, 64),
            nn.ReLU(),
            nn.Linear(64, num_acciones_salida)
        )

    def forward(self, x):
        return self.modelo(x)

    def seleccionar_accion(self, estado):
        estado_tensor = torch.tensor(estado.flatten(), dtype=torch.float32).unsqueeze(0)
        logits = self.forward(estado_tensor)
        accion = torch.argmax(logits, dim=1).item()
        return accion

# ===================================================================
# 🔹 FUNCIONES DE UTILIDAD PARA NES
# ===================================================================

def obtener_parametros_flat(modelo):
    """Devuelve todos los parámetros de la red como un solo vector plano."""
    return torch.cat([p.data.view(-1) for p in modelo.parameters()])

def asignar_parametros_flat(modelo, vector_parametros):
    """Asigna un vector plano de parámetros a la red."""
    indice = 0
    for p in modelo.parameters():
        cantidad_elementos = p.numel()
        porcion = vector_parametros[indice:indice + cantidad_elementos]
        p.data.copy_(porcion.view(p.size()))
        indice += cantidad_elementos

def evaluar_politica(env, politica, render=False):
    """
    Ejecuta un episodio completo usando una política dada.
    Retorna la recompensa total acumulada.
    """
    env.reset()
    recompensa_total = 0

    # Acción inicial forzada: disparar (índice 5 en MinAtar Breakout)
    recompensa, terminado = env.act(5)
    recompensa_total += recompensa
    estado = env.state()

    # Ejecutar el episodio
    while not terminado:
        accion = politica.seleccionar_accion(estado)
        recompensa, terminado = env.act(accion)
        recompensa_total += recompensa
        estado = env.state()

        if render:
            print(env.display())

    return recompensa_total

# ===================================================================
# 🔹 ENTRENAMIENTO CON NES
# ===================================================================

# Hiperparámetros
episodios_entrenamiento = 1000     # Cuántas iteraciones NES realizar
num_perturbaciones = 100           # Cuántas muestras generar por episodio
desviacion_ruido = 0.1             # Magnitud del ruido (σ)
tasa_aprendizaje = 0.01            # Step size (α)

# Inicializar la política
politica = PoliticaNES(tamano_estado_flat, num_acciones)

# Historial de recompensas para graficar
recompensas_promedio_por_episodio = []

# Bucle de entrenamiento NES
for ep in range(episodios_entrenamiento):
    # Obtener vector actual de parámetros θ
    theta_actual = obtener_parametros_flat(politica)

    # Generar perturbaciones y evaluar
    lista_perturbaciones = []
    lista_recompensas = []

    for _ in range(num_perturbaciones):
        ruido = torch.randn_like(theta_actual)
        theta_alterado = theta_actual + desviacion_ruido * ruido
        asignar_parametros_flat(politica, theta_alterado)

        recompensa = evaluar_politica(entorno, politica)
        lista_perturbaciones.append(ruido)
        lista_recompensas.append(recompensa)

    # Guardar recompensa promedio antes de normalizar
    recompensa_promedio_cruda = sum(lista_recompensas) / num_perturbaciones
    recompensas_promedio_por_episodio.append(recompensa_promedio_cruda)

    # Normalizar recompensas para mejor estabilidad numérica
    recompensas_tensor = torch.tensor(lista_recompensas, dtype=torch.float32)
    recompensas_normalizadas = (recompensas_tensor - recompensas_tensor.mean()) / (recompensas_tensor.std() + 1e-8)

    # Estimar el gradiente de la recompensa esperada
    gradiente_estimado = torch.zeros_like(theta_actual)
    for R, eps in zip(recompensas_normalizadas, lista_perturbaciones):
        gradiente_estimado += R * eps
    gradiente_estimado /= (num_perturbaciones * desviacion_ruido)

    # Actualizar los parámetros en la dirección estimada
    theta_actual += tasa_aprendizaje * gradiente_estimado
    asignar_parametros_flat(politica, theta_actual)

    print(f"Episodio {ep+1}/{episodios_entrenamiento} - Recompensa promedio: {recompensa_promedio_cruda:.2f}")

# ===================================================================
# 🔹 VISUALIZACIÓN DE RESULTADOS
# ===================================================================

plt.plot(recompensas_promedio_por_episodio)
plt.xlabel("Episodio")
plt.ylabel("Recompensa promedio")
plt.title("Rendimiento de NES en Breakout")
plt.grid()
plt.show()


