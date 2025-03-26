import gymnasium as gym  # Importar Gymnasium para simular entornos de aprendizaje por refuerzo
import numpy as np  # Importar NumPy para cálculos numéricos eficientes
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar resultados
import random  # Importar Random para selección aleatoria en la planificación
import pandas as pd  # Importar Pandas para manejar y visualizar datos en formato tabular

# Configuración del entorno
env = gym.make("MountainCar-v0")  # Crear el entorno MountainCar de Gymnasium

# Hiperparámetros del algoritmo Dyna-Q
alpha = 0.1           # Tasa de aprendizaje (qué tan rápido actualiza los valores de Q)
gamma = 0.99          # Factor de descuento (cuánto valora las recompensas futuras)
epsilon = 1.0         # Probabilidad de exploración inicial (para la estrategia ε-greedy)
num_episodios = 3000  # Número de episodios de entrenamiento
n_planificación = 50  # Número de iteraciones de planificación en Dyna-Q
epsilon_decay = 0.999 # Tasa de reducción de ε (para disminuir exploración con el tiempo)

# Discretización del espacio de estados
num_bins = (40, 40)  # Número de divisiones en la discretización del espacio de estados
limites_estado = list(zip(env.observation_space.low, env.observation_space.high))  # Obtener límites del espacio de estados
limites_estado[1] = (-0.07, 0.07)  # Ajustar el límite de velocidad para evitar problemas de borde

# Función para convertir un estado continuo en un estado discreto
def discretizar_estado(estado):
    return tuple(
        np.digitize(estado[i], np.linspace(limites_estado[i][0], limites_estado[i][1], num_bins[i] - 1)) 
        for i in range(2)
    )

# Inicialización de la tabla Q con valores pequeños aleatorios
Q = np.random.uniform(low=-0.1, high=0.1, size=(num_bins[0], num_bins[1], env.action_space.n))

# Modelo de transición (almacenará experiencias en formato (s, a) → [(r, s')])
modelo = {}

# Función para seleccionar una acción usando una estrategia ε-greedy
def seleccionar_accion(estado, epsilon):
    if np.random.rand() < epsilon:  # Explorar con probabilidad ε
        return env.action_space.sample()  # Seleccionar acción aleatoria
    else:
        return np.argmax(Q[estado])  # Explotar el mejor valor de Q conocido

# Entrenamiento con Dyna-Q
recompensas_por_episodio = []  # Lista para almacenar la recompensa total por episodio

for episodio in range(num_episodios):
    estado_continuo, _ = env.reset(seed=None)  # Reiniciar el entorno
    estado = discretizar_estado(estado_continuo)  # Convertir estado a su versión discretizada
    recompensa_total = 0  # Inicializar acumulador de recompensa del episodio
    terminado = False  # Variable para indicar si el episodio terminó

    while not terminado:
        accion = seleccionar_accion(estado, epsilon)  # Seleccionar acción con ε-greedy
        nuevo_estado_continuo, recompensa, es_terminal, es_truncado, _ = env.step(accion)  # Ejecutar la acción en el entorno
        nuevo_estado = discretizar_estado(nuevo_estado_continuo)  # Discretizar el nuevo estado observado

        # Modificar la recompensa para incentivar el progreso
        recompensa += 100 * (nuevo_estado_continuo[0] - estado_continuo[0])

        # Actualizar la tabla Q con la regla de aprendizaje de Q-learning
        mejor_Q = np.max(Q[nuevo_estado]) if not es_terminal else 0  # Obtener el mejor valor Q del siguiente estado
        Q[estado][accion] += alpha * (recompensa + gamma * mejor_Q - Q[estado][accion])  # Actualización de Q

        # Almacenar la transición en el modelo Dyna-Q
        if (estado, accion) not in modelo:
            modelo[(estado, accion)] = []
        modelo[(estado, accion)].append((recompensa, nuevo_estado))

        # Mantener solo las últimas 5 transiciones por estado-acción para evitar sobrecarga
        if len(modelo[(estado, accion)]) > 5:
            modelo[(estado, accion)].pop(0)

        # Realizar planificación con estados diversos
        estados_sample = random.sample(list(modelo.keys()), min(len(modelo), n_planificación))
        for s_a in estados_sample:
            s, a = s_a
            r, s_prime = random.choice(modelo[s_a])  # Elegir una transición aleatoria almacenada
            mejor_Q_modelo = np.max(Q[s_prime]) if s_prime in modelo else 0  # Obtener el mejor Q del estado siguiente
            Q[s][a] += alpha * (r + gamma * mejor_Q_modelo - Q[s][a])  # Actualizar Q basado en la planificación

        # Avanzar al siguiente estado
        estado_continuo = nuevo_estado_continuo
        estado = nuevo_estado
        recompensa_total += recompensa  # Acumular la recompensa total del episodio
        terminado = es_terminal or es_truncado  # Verificar si el episodio terminó

    recompensas_por_episodio.append(recompensa_total)  # Almacenar la recompensa del episodio

    # Reducir ε gradualmente para favorecer explotación en etapas posteriores
    epsilon = max(0.01, epsilon * epsilon_decay)

    # Depuración: Mostrar progreso cada 500 episodios
    if (episodio + 1) % 500 == 0:
        print(f"Episodio {episodio + 1}: Recompensa Total = {recompensa_total}")
        print(f"Ejemplo Q[estado aleatorio]: {Q[random.randint(0, num_bins[0]-1), random.randint(0, num_bins[1]-1)]}")
        print(f"Total de transiciones en modelo: {len(modelo)}")

env.close()

# Graficar la evolución de la recompensa por episodio
plt.figure(figsize=(10, 5))
plt.plot(recompensas_por_episodio, label="Recompensa por episodio", alpha=0.7)
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Evolución de la recompensa con Dyna-Q")
plt.legend()
plt.show()

# Evaluación del agente entrenado con 10 experimentos
num_experimentos = 10
recompensas_finales = []

for experimento in range(num_experimentos):
    estado_continuo, _ = env.reset(seed=None)
    recompensa_total = 0
    terminado = False

    while not terminado:
        accion = seleccionar_accion(estado, epsilon=0)  # Política óptima (sin exploración)
        nuevo_estado_continuo, recompensa, es_terminal, es_truncado, _ = env.step(accion)
        terminado = es_terminal or es_truncado
        recompensa_total += recompensa
        estado_continuo = nuevo_estado_continuo

    recompensas_finales.append(recompensa_total)

# Cálculo de métricas estadísticas
media_recompensas = np.mean(recompensas_finales)
desviacion_recompensas = np.std(recompensas_finales)

# Crear un DataFrame con los resultados
resultados_df = pd.DataFrame({
    "Algoritmo": ["Dyna-Q"] * num_experimentos,
    "Problema": ["MountainCar-v0"] * num_experimentos,
    "Experimento": np.arange(1, num_experimentos + 1),
    "Recompensa Total": recompensas_finales
})

# Imprimir el reporte de evaluación
print("\nReporte de Evaluación:")
print(resultados_df.to_string(index=False))
print(f"\nMedia de recompensa: {media_recompensas:.2f}")
print(f"Desviación estándar de recompensa: {desviacion_recompensas:.2f}")
