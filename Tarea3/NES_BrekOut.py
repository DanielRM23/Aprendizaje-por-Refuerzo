# 🔹 Importación de librerías
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from minatar import Environment


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



import torch
import torch.nn as nn

# Definimos la clase de la red neuronal para la política
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()  # Llama al constructor de nn.Module

        # Creamos un modelo secuencial con:
        # - Una capa lineal que va de input_dim a 64 neuronas
        # - Una función de activación ReLU
        # - Otra capa lineal que va de 64 neuronas a output_dim (una por acción posible)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # Capa densa de entrada
            nn.ReLU(),                 # Activación no lineal
            nn.Linear(64, output_dim)  # Capa de salida con tantas neuronas como acciones
        )

    # Método que define cómo se pasa un input por la red
    def forward(self, x):
        # Simplemente pasa el input a través del modelo secuencial
        return self.model(x)

    # Método para elegir una acción dado un estado
    def act(self, state):
        # Aplana el estado (por si viene en forma 2D o 3D)
        # Lo convierte en tensor float32 y le agrega una dimensión para representar un batch (batch_size = 1)
        state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)  # shape: (1, input_dim)

        # Obtiene los valores de salida (logits) para cada acción
        logits = self.forward(state)

        # Toma la acción con el valor más alto (greedy)
        action = torch.argmax(logits, dim=1).item()

        # Devuelve la acción como un número entero
        return action
