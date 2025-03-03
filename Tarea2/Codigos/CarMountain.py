import gymnasium as gym
import matplotlib.pyplot as plt

# Crear el entorno con renderizado
entorno = gym.make("MountainCar-v0", render_mode="rgb_array")

# Reiniciar el entorno
estado, info = entorno.reset(seed=123)

# Tomar una acción (ejemplo: acelerar a la derecha)
accion = 2  # 0 = Izquierda, 1 = Nada, 2 = Derecha
nuevo_estado, recompensa, terminado, truncado, _ = entorno.step(accion)

# Obtener una imagen del entorno
imagen = entorno.render()

# Mostrar la imagen del entorno
plt.imshow(imagen)
plt.axis("off")  # Ocultar ejes
plt.title("Vista de MountainCar después de una acción")
plt.show()

# Cerrar el entorno
entorno.close()
