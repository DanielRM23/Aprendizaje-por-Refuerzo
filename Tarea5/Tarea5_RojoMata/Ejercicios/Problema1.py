# Importamos las librerías necesarias
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Creamos una clase que adapta CartPole a un entorno multiobjetivo
# Lo que se hace es agregar una nueva recompensa: la inclinación del palo 
class CartPoleMultiobjetivo(gym.Wrapper):
    def __init__(self, entorno):
        # Inicializamos el wrapper con el entorno base
        super().__init__(entorno)

    def step(self, accion):
        # Ejecutamos la acción en el entorno base y obtenemos la transición
        observacion, recompensa, terminado, truncado, info = self.env.step(accion)

        # Primer objetivo: recompensa original (tiempo de supervivencia)
        recompensa_1 = recompensa

        # Segundo objetivo: castigo por la inclinación del palo (ángulo distinto de 0)
        angulo = observacion[2]
        recompensa_2 = -abs(angulo)

        # Combinamos ambas recompensas en un vector
        recompensa_vector = np.array([recompensa_1, recompensa_2])

        # Retornamos la observación, el vector de recompensas y los indicadores de finalización
        return observacion, recompensa_vector, terminado, truncado, info

    def reset(self, **kwargs):
        # Reiniciamos el entorno
        return self.env.reset(**kwargs)


# Función para simular un episodio con el entorno dado
def simular_episodio(entorno, es_multiobjetivo=True):
    observacion, _ = entorno.reset()
    terminado = False
    historial_r1, historial_r2 = [], []

    while not terminado:
        # Elegimos una acción aleatoria
        accion = entorno.action_space.sample()
        observacion, recompensas, fin, truncado, _ = entorno.step(accion)

        if es_multiobjetivo:
            historial_r1.append(recompensas[0])
            historial_r2.append(recompensas[1])
        else:
            historial_r1.append(recompensas)
            historial_r2.append(0)  # No hay segunda recompensa

        terminado = fin or truncado

    # Devolvemos las recompensas acumuladas
    return np.cumsum(historial_r1), np.cumsum(historial_r2)


# Bloque principal de ejecución
if __name__ == "__main__":
    # Simulación del entorno original (monoobjetivo)
    entorno_original = gym.make("CartPole-v1")
    r1_mono, r2_mono = simular_episodio(entorno_original, es_multiobjetivo=False)

    # Simulación del entorno adaptado (multiobjetivo)
    entorno_multi = CartPoleMultiobjetivo(gym.make("CartPole-v1"))
    r1_multi, r2_multi = simular_episodio(entorno_multi, es_multiobjetivo=True)

    # Graficamos los resultados
    plt.plot(r1_mono, r2_mono, label="Mono-objetivo", linestyle='--')
    plt.plot(r1_multi, r2_multi, label="Multi-objetivo", linestyle=':', color="magenta")
    plt.xlabel("Duración del episodio (pasos acumulados)")
    plt.ylabel("Recompensa acumulada: castigo por inclinación")
    plt.title("Comparación entre entornos")
    plt.legend()
    plt.grid(True)
    plt.savefig("Problema1.png")
    plt.show()
