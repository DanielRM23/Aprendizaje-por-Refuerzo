import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

# Configuración del entorno
entorno = gym.make("MountainCar-v0", render_mode="rgb_array")

# Hiperparámetros del modelo
tasa_aprendizaje_actor = 0.1  # Velocidad de aprendizaje del actor (θ)
tasa_aprendizaje_critico = 0.1  # Velocidad de aprendizaje del crítico (w)
factor_descuento = 0.99  # Descuento para futuras recompensas (γ)
num_episodios = 5000  # Número de episodios de entrenamiento
epsilon = 0.1  # Exploración inicial en la estrategia ε-greedy

# Inicialización de pesos para la aproximación lineal
num_caracteristicas = 4  # Cantidad de características usadas en la representación del estado
num_acciones = entorno.action_space.n  # Número de acciones posibles en el entorno (izquierda, nada, derecha)

# Pesos del actor (θ) y crítico (w) inicializados aleatoriamente con valores pequeños
pesos_actor = np.random.rand(num_caracteristicas, num_acciones) * 0.01
pesos_critico = np.random.rand(num_caracteristicas) * 0.01

# Función para transformar el estado en características útiles para el modelo
def transformar_estado(estado):
    posicion, velocidad = estado
    return np.array([
        posicion,  # Posición actual del auto
        velocidad,  # Velocidad actual
        posicion ** 2,  # Posición elevada al cuadrado para capturar relaciones no lineales
        velocidad ** 2  # Velocidad elevada al cuadrado para capturar relaciones no lineales
    ])

# Función de selección de acción usando política ε-greedy
def seleccionar_accion(estado, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_acciones)  # Exploración: elegimos una acción aleatoria
    else:
        caracteristicas_estado = transformar_estado(estado)
        preferencias = np.dot(caracteristicas_estado, pesos_actor)  # Cálculo de preferencias de acción
        return np.argmax(preferencias)  # Acción con la mayor probabilidad

# Función de valor del estado según la aproximación lineal
def valor_estado(estado):
    return np.dot(transformar_estado(estado), pesos_critico)  # Estimación de V(s) usando pesos del crítico

# Entrenamiento Actor-Crítico
recompensas_por_episodio = []

for episodio in range(num_episodios):
    estado, _ = entorno.reset(seed=None)  # Reiniciamos el entorno
    terminado = False  # Variable para controlar el fin del episodio
    I = 1  # Factor de descuento multiplicativo
    recompensa_total = 0  # Acumulador de recompensas del episodio

    while not terminado:
        accion = seleccionar_accion(estado, epsilon)  # Seleccionar acción basada en la política ε-greedy
        nuevo_estado, recompensa, es_terminal, es_truncado, _ = entorno.step(accion)  # Ejecutar acción
        terminado = es_terminal or es_truncado  # Verificar si el episodio termina

        # Modificación de la recompensa: Se otorga un refuerzo por avanzar en la posición
        recompensa += 100 * (nuevo_estado[0] - estado[0])  # Incentivar el avance del auto

        recompensa_total += recompensa  # Acumular recompensa total del episodio

        # Cálculo del error TD (δ)
        objetivo_td = recompensa + factor_descuento * valor_estado(nuevo_estado) * (not es_terminal)
        error_td = objetivo_td - valor_estado(estado)

        # Normalización del TD-error para evitar gradientes explosivos
        error_td = np.clip(error_td, -1, 1)

        # Actualización de los pesos del crítico usando gradiente descendente
        pesos_critico += tasa_aprendizaje_critico * error_td * transformar_estado(estado)

        # Actualización del actor
        probabilidades_acciones = np.exp(np.dot(transformar_estado(estado), pesos_actor))  # Cálculo de probabilidades Softmax
        probabilidades_acciones /= np.sum(probabilidades_acciones)  # Normalización

        gradiente_ln_pi = -probabilidades_acciones  # Gradiente del logaritmo de la política
        gradiente_ln_pi[accion] += 1  # Ajustamos solo la acción seleccionada

        # Aplicamos el ajuste al actor
        pesos_actor += tasa_aprendizaje_actor * I * error_td * np.outer(transformar_estado(estado), gradiente_ln_pi)

        I *= factor_descuento  # Ajustamos el factor de descuento
        estado = nuevo_estado  # Actualizamos el estado

    recompensas_por_episodio.append(recompensa_total)  # Guardamos la recompensa total del episodio

    # Imprimir progreso cada 500 episodios
    if (episodio + 1) % 500 == 0:
        print(f"Episodio {episodio + 1}: Recompensa Total = {recompensa_total}")

    # Reducción progresiva de la exploración (decay de ε)
    epsilon = max(0.01, epsilon * 0.995)

entorno.close()  # Cerramos el entorno después del entrenamiento

# Graficar la evolución de la recompensa a lo largo del entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(recompensas_por_episodio, label="Recompensa por episodio", alpha=0.7)
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Evolución de la recompensa durante el entrenamiento")
plt.legend()
plt.show()

# Evaluación de la política aprendida con 10 experimentos
num_experimentos = 10
recompensas_finales = []

for experimento in range(num_experimentos):
    estado, _ = entorno.reset(seed=None)  # Reiniciar entorno para prueba
    recompensa_total = 0
    terminado = False

    while not terminado:
        accion = seleccionar_accion(estado, epsilon=0)  # Seleccionar acción óptima (sin exploración)
        nuevo_estado, recompensa, es_terminal, es_truncado, _ = entorno.step(accion)  # Ejecutar acción
        terminado = es_terminal or es_truncado  # Verificar si el episodio termina
        recompensa_total += recompensa  # Acumular recompensa total
        estado = nuevo_estado  # Actualizar estado

    recompensas_finales.append(recompensa_total)  # Guardar resultado de la prueba

# Cálculo de métricas estadísticas
media_recompensas = np.mean(recompensas_finales)  # Media de recompensas en evaluación
desviacion_recompensas = np.std(recompensas_finales)  # Desviación estándar

# Mostrar resultados en consola
print(f"\nResultados tras {num_experimentos} ejecuciones:")
print(f"Media de recompensa: {media_recompensas}")
print(f"Desviación estándar de recompensa: {desviacion_recompensas}")

# Crear DataFrame para mostrar resultados de la evaluación
resultados_df = pd.DataFrame({
    "Experimento": np.arange(1, num_experimentos + 1),
    "Recompensa Total": recompensas_finales
})

# Imprimir la tabla con los resultados finales
print("\nResultados de Evaluación:")
print(resultados_df.to_string(index=False))
