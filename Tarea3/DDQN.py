from minatar import Environment
import numpy as np

# 🔹 Inicializar el entorno de Breakout en MinAtar
entorno_juego = Environment("breakout")  # Cargar el entorno
num_acciones_posibles = entorno_juego.num_actions()  # Número de acciones posibles
forma_estado_juego = entorno_juego.state_shape()  # Forma del estado
dimension_estado = np.prod(forma_estado_juego)  # Dimensión total del estado si se aplana

# 🔹 Imprimir información del entorno
print(f"Número de acciones posibles: {num_acciones_posibles}")
print(f"Forma del estado del juego: {forma_estado_juego}")

# 🔹 Obtener el estado inicial
estado_actual = entorno_juego.state()
print(f"Dimensión del estado inicial: {estado_actual.shape}")

# 🔹 Realizar una acción aleatoria en el entorno
entorno_juego.reset()  # Reinicia el entorno
accion_ejecutada = np.random.randint(num_acciones_posibles)  # Selecciona una acción aleatoria
recompensa_obtenida, episodio_terminado = entorno_juego.act(accion_ejecutada)  # Aplica la acción

print(f"Acción ejecutada: {accion_ejecutada}, Recompensa: {recompensa_obtenida}, ¿Episodio terminado? {episodio_terminado}")




import torch  # Biblioteca base de PyTorch
import torch.nn as nn  # Módulos para definir redes neuronales
import torch.optim as optim  # Optimización para entrenar la red

# 🔹 Definir la red neuronal para Deep Q-Network (DQN)
class RedQ(nn.Module):  # La clase hereda de nn.Module (base de redes en PyTorch)
    def __init__(self, forma_entrada, num_acciones):
        super(RedQ, self).__init__()  # Llama al constructor de nn.Module

        # 🔹 Capa convolucional 1: Detecta patrones espaciales en la imagen
        self.capa_conv1 = nn.Conv2d(
            in_channels=forma_entrada[2],  # Número de canales de entrada (4 en MinAtar), recuerda; forma_entrada = [10,10,4]
            out_channels=32,  # Número de filtros aprendidos (32 detectores de características)
            kernel_size=3,  # Tamaño del filtro 3x3
            stride=1,  # Paso del filtro (1 = se mueve un píxel a la vez)
            padding=1  # Se añaden ceros alrededor para mantener el tamaño de salida
        )
        self.activacion1 = nn.ReLU()  # Activación ReLU para introducir no linealidad

        # 🔹 Capa convolucional 2: Extrae características más profundas
        self.capa_conv2 = nn.Conv2d(
            in_channels=32,  # Usa las 32 características detectadas en la capa anterior
            out_channels=64,  # Ahora detecta 64 patrones diferentes
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.activacion2 = nn.ReLU()

        # 🔹 Capa completamente conectada (Dense)
        self.aplanado = nn.Flatten()  # Aplana la salida de las capas convolucionales

        # 🔹 Primera capa densa: Reduce la dimensionalidad antes de la salida final
        self.capa_densa1 = nn.Linear(
            64 * forma_entrada[0] * forma_entrada[1],  # Número de entradas (64 mapas de características de 10x10)
            256  # Número de neuronas ocultas
        )
        self.activacion3 = nn.ReLU()

        # 🔹 Capa de salida: Predice valores Q para cada acción posible
        self.capa_salida = nn.Linear(256, num_acciones)  # La salida tiene el tamaño de las acciones posibles

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


# 🔹 Configuración del dispositivo (usa GPU si está disponible)
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Crear la red principal Q-Network
red_principal = RedQ(forma_entrada=[10, 10, 4], num_acciones=num_acciones_posibles).to(dispositivo)

# 🔹 Crear la red objetivo Target Q-Network (igual a la red principal inicialmente)
red_objetivo = RedQ(forma_entrada=[10, 10, 4], num_acciones=num_acciones_posibles).to(dispositivo)
red_objetivo.load_state_dict(red_principal.state_dict())  # Copia los pesos de la red principal a la red objetivo
red_objetivo.eval()  # La red objetivo no se entrena directamente

# 🔹 Definir el optimizador Adam para actualizar la red principal
optimizador = optim.Adam(red_principal.parameters(), lr=0.001)  # Tasa de aprendizaje de 0.001

# 🔹 Definir la función de pérdida (MSE: Error cuadrático medio)
funcion_perdida = nn.MSELoss()  # Se usa para comparar los valores Q estimados con los valores objetivo

print("Modelo creado y listo para entrenar.")
