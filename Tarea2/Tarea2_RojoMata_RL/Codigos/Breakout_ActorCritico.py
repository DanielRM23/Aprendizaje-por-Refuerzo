from minatar import Environment
import numpy as np
import matplotlib.pyplot as plt

# Crear el entorno de MinAtar para el juego Breakout
entorno = Environment("breakout")
num_acciones = entorno.num_actions()
forma_estado = entorno.state_shape()
dim_estado = np.prod(forma_estado)

# Inicializar pesos de la política y la función de valor
pesos_politica = np.random.uniform(-0.01, 0.01, (num_acciones, dim_estado))
pesos_valor = np.random.uniform(-0.01, 0.01, dim_estado)

# Hiperparámetros del algoritmo
alpha_politica = 0.002  # Tasa de aprendizaje de la política
alpha_valor = 0.01  # Tasa de aprendizaje de la función de valor
descuento_gamma = 0.99  # Factor de descuento
num_episodios = 5000  # Número de episodios de entrenamiento

def preprocesar_estado(estado):
    """Convierte la matriz del estado en un vector y escala valores entre 0 y 1."""
    return estado.flatten() / 255.0

def obtener_probabilidades_accion(estado):
    """Calcula las probabilidades de cada acción utilizando una política parametrizada."""
    logits = np.dot(pesos_politica, estado)  # Producto punto entre pesos y estado
    logits = np.clip(logits, -10, 10)  # Evita valores extremos en los logits
    exp_logits = np.exp(logits - np.max(logits))  # Normaliza para estabilidad numérica
    exp_logits = np.where(exp_logits > 1e-10, exp_logits, 1e-10)  # Evita valores pequeños
    return exp_logits / np.sum(exp_logits)  # Obtiene las probabilidades normalizadas

def seleccionar_accion(estado):
    """Selecciona una acción basada en la política actual."""
    probabilidades = obtener_probabilidades_accion(estado)
    return np.random.choice(len(probabilidades), p=probabilidades)

def actualizar_politica(estado, accion, delta, factor_importancia):
    """Actualiza los pesos de la política usando el gradiente de la ventaja."""
    global pesos_politica
    probabilidades = obtener_probabilidades_accion(estado)
    gradiente = -probabilidades  # Gradiente negativo para todas las acciones
    gradiente[accion] += 1  # Refuerza la acción tomada
    pesos_politica += alpha_politica * factor_importancia * delta * np.outer(gradiente, estado)

def predecir_valor(estado):
    """Predice el valor del estado usando la función de valor."""
    return np.dot(pesos_valor, estado)

def actualizar_funcion_valor(estado, delta):
    """Actualiza los pesos de la función de valor usando el error TD."""
    global pesos_valor
    pesos_valor += alpha_valor * delta * estado

# Almacenar recompensas por episodio para graficar
recompensas_por_episodio = []

# Ciclo principal de entrenamiento
for episodio in range(num_episodios):
    entorno.reset()  # Reiniciar el entorno
    estado = preprocesar_estado(entorno.state())  # Obtener estado inicial
    factor_importancia = 1  # Inicializar factor de descuento acumulado
    terminado = False
    recompensa_total = 0
    
    while not terminado:
        # Seleccionar acción según la política con exploración epsilon-greedy
        epsilon = max(0.4 - episodio / (num_episodios / 2), 0.05)
        if np.random.rand() < epsilon:
            accion = np.random.choice(num_acciones)  # Acción aleatoria
        else:
            accion = seleccionar_accion(estado)  # Acción basada en la política
        
        recompensa, terminado = entorno.act(accion)  # Ejecutar acción y obtener recompensa
        recompensa = max(recompensa, -10)  # Limitar recompensas negativas
        siguiente_estado = preprocesar_estado(entorno.state())  # Obtener el siguiente estado
        recompensa_total += recompensa  # Acumular recompensa
        
        # Calcular el error TD (Diferencia Temporal)
        valor_actual = predecir_valor(estado)
        valor_siguiente = predecir_valor(siguiente_estado) if not terminado else 0.0
        delta = np.clip(recompensa + descuento_gamma * valor_siguiente - valor_actual, -5, 5)
        
        # Actualizar función de valor y política
        actualizar_funcion_valor(estado, delta)
        actualizar_politica(estado, accion, delta, factor_importancia)
        
        # Actualizar factor de importancia
        factor_importancia *= descuento_gamma
        estado = siguiente_estado
    
    recompensas_por_episodio.append(recompensa_total)
    print(f"Episodio {episodio+1}/{num_episodios} completado - Recompensa Total: {recompensa_total}")

entorno.close_display()

# Graficar recompensa obtenida por episodio
plt.plot(range(num_episodios), recompensas_por_episodio)
plt.xlabel("Episodio")
plt.ylabel("Recompensa acumulada")
plt.title("Recompensa obtenida por episodio")
plt.show()

# Evaluar la política entrenada con 10 experimentos
num_experimentos = 10
recompensas_experimentos = []

for _ in range(num_experimentos):
    entorno.reset()
    estado = preprocesar_estado(entorno.state())
    terminado = False
    recompensa_total = 0
    
    while not terminado:
        accion = seleccionar_accion(estado)  # Seleccionar acción basada en la política
        recompensa, terminado = entorno.act(accion)  # Ejecutar acción y obtener recompensa
        recompensa_total += recompensa  # Acumular recompensa
        estado = preprocesar_estado(entorno.state())
    
    recompensas_experimentos.append(recompensa_total)

# Calcular media y desviación estándar de la recompensa
recompensa_media = np.mean(recompensas_experimentos)
desviacion_recompensa = np.std(recompensas_experimentos)

print(f"Recompensa media después de 10 experimentos: {recompensa_media:.2f}")
print(f"Desviación estándar de la recompensa: {desviacion_recompensa:.2f}")