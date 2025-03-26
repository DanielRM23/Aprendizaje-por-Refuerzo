# Importamos las librerías necesarias
import gymnasium as gym  # Importar el entorno de OpenAI Gymnasium
import numpy as np  # Importar NumPy para cálculos numéricos
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar
import random  # Importar random para seleccionar experiencias aleatorias

# Crear el entorno CartPole
entorno = gym.make("CartPole-v1")

# Obtener dimensiones del espacio de estado y acción
dim_estado = entorno.observation_space.shape[0]  # Número de características del estado
dim_accion = entorno.action_space.n  # Número de acciones disponibles

# Hiperparámetros del algoritmo
alpha = 0.1  # Tasa de aprendizaje (learning rate)
gamma = 0.99  # Factor de descuento (discount factor)
epsilon = 0.1  # Probabilidad de exploración en la política epsilon-greedy
n_planificacion = 10  # Número de iteraciones de planificación por paso
num_bins = 10  # Número de divisiones para discretizar el estado

# Definir los límites del espacio de observaciones
limites = list(zip(entorno.observation_space.low, entorno.observation_space.high))
limites[1] = (-2.4, 2.4)  # Limitar el rango de la posición del carrito
limites[3] = (-2.0, 2.0)  # Limitar el rango de la velocidad angular del palo

# Crear la tabla Q con estados discretizados
Q = np.zeros((num_bins,) * dim_estado + (dim_accion,))  # Inicializar la tabla Q con ceros
tabla_modelo = {}  # Diccionario para almacenar el modelo del entorno

# Función para discretizar el estado continuo
def discretizar_estado(estado):
    """Convierte un estado continuo en un índice discreto para la tabla Q."""
    estado_discreto = []
    for i, (valor, (min_val, max_val)) in enumerate(zip(estado, limites)):
        if valor < min_val:
            indice = 0  # Si el valor es menor que el mínimo, asignarlo al primer bin
        elif valor > max_val:
            indice = num_bins - 1  # Si el valor es mayor que el máximo, asignarlo al último bin
        else:
            indice = int(((valor - min_val) / (max_val - min_val)) * (num_bins - 1))  # Normalización
        estado_discreto.append(indice)
    return tuple(estado_discreto)  # Retornar el estado discretizado como una tupla

# Función para seleccionar una acción usando epsilon-greedy
def epsilon_greedy(estado, Q, epsilon):
    """Selecciona una acción usando política epsilon-greedy."""
    if np.random.rand() < epsilon:  # Con probabilidad epsilon, elige una acción aleatoria
        return np.random.choice(dim_accion)
    else:  # De lo contrario, elige la mejor acción conocida según la tabla Q
        return np.argmax(Q[estado])

# Función para actualizar el modelo del entorno
def actualizar_modelo(estado, accion, recompensa, siguiente_estado):
    """Guarda la transición en el modelo."""
    tabla_modelo[(estado, accion)] = (recompensa, siguiente_estado)

# Función para realizar planificación en Dyna-Q
def planificar(Q, tabla_modelo, n_planificacion, alpha, gamma):
    """Realiza planificación basada en las experiencias almacenadas."""
    for _ in range(n_planificacion):  # Repetir planificación n veces
        if len(tabla_modelo) == 0:  # Si no hay experiencias almacenadas, salir
            break
        (estado, accion), (recompensa, siguiente_estado) = random.choice(list(tabla_modelo.items()))  # Elegir una experiencia al azar
        Q[estado][accion] += alpha * (recompensa + gamma * np.max(Q[siguiente_estado]) - Q[estado][accion])  # Actualizar Q

# Función principal para entrenar el agente con Dyna-Q
def dyna_q():
    """Entrena un agente con Dyna-Q en el entorno CartPole."""
    recompensas_por_paso = []  # Lista para almacenar recompensas acumuladas
    
    for episodio in range(1000):  # Ejecutar 1000 episodios de entrenamiento
        estado, _ = entorno.reset()  # Reiniciar el entorno
        estado = discretizar_estado(estado)  # Discretizar el estado
        recompensa_total = 0  # Inicializar la recompensa acumulada
        terminado = False  # Indicador de si el episodio terminó
        
        while not terminado:  # Mientras el episodio no termine
            accion = epsilon_greedy(estado, Q, epsilon)  # Elegir acción con política epsilon-greedy
            siguiente_estado, recompensa, terminado_env, truncado, _ = entorno.step(accion)  # Ejecutar acción
            terminado = terminado_env or truncado  # Verificar si el episodio termina
            siguiente_estado = discretizar_estado(siguiente_estado)  # Discretizar el siguiente estado
            recompensa_total += recompensa  # Acumular la recompensa
            
            # Actualizar Q-Table con la ecuación de actualización de Q-learning
            Q[estado][accion] += alpha * (recompensa + gamma * np.max(Q[siguiente_estado]) - Q[estado][accion])
            
            # Almacenar la experiencia en el modelo
            actualizar_modelo(estado, accion, recompensa, siguiente_estado)
            
            # Realizar planificación con experiencias almacenadas
            planificar(Q, tabla_modelo, n_planificacion, alpha, gamma)
            
            estado = siguiente_estado  # Actualizar el estado actual
        
        recompensas_por_paso.append(recompensa_total)  # Guardar la recompensa total del episodio
        if episodio % 100 == 0:
            print(f"Episodio {episodio} completado.")  # Imprimir cada 100 episodios
    
    # Graficar resultados
    plt.plot(range(len(recompensas_por_paso)), recompensas_por_paso)
    plt.xlabel("Pasos")  # Etiqueta del eje X
    plt.ylabel("Recompensa Promedio")  # Etiqueta del eje Y
    plt.title("Entrenamiento con Dyna-Q en CartPole")  # Título del gráfico
    plt.show()  # Mostrar gráfico
    
    return Q  # Retornar la tabla Q entrenada

# Función para evaluar la política aprendida
def evaluar_politica(Q, num_ejecuciones=10):
    """Evalúa la política aprendida usando la Q-Table."""
    recompensas = []  # Lista para almacenar recompensas obtenidas
    
    for _ in range(num_ejecuciones):  # Ejecutar varias evaluaciones
        estado, _ = entorno.reset()  # Reiniciar el entorno
        estado = discretizar_estado(estado)  # Discretizar el estado
        terminado = False  # Indicador de si el episodio terminó
        recompensa_total = 0  # Inicializar la recompensa acumulada
        
        while not terminado:
            accion = np.argmax(Q[estado])  # Elegir la mejor acción basada en la Q-Table
            estado, recompensa, terminado_env, truncado, _ = entorno.step(accion)  # Ejecutar acción
            terminado = terminado_env or truncado  # Verificar si el episodio terminó
            estado = discretizar_estado(estado)  # Discretizar el siguiente estado
            recompensa_total += recompensa  # Acumular recompensa
        
        recompensas.append(recompensa_total)  # Guardar la recompensa obtenida en la evaluación
    
    # Calcular estadísticas de evaluación
    recompensa_media = np.mean(recompensas)  # Promedio de recompensas obtenidas
    desviacion_recompensa = np.std(recompensas)  # Desviación estándar de las recompensas
    
    # Mostrar resultados de evaluación
    print("\nResultados de Evaluación:")
    print("-----------------------------------")
    print(f"| Recompensa Media: {recompensa_media:.2f} | Desviación Estándar: {desviacion_recompensa:.2f} |")
    print("-----------------------------------")
    
    return recompensa_media, desviacion_recompensa  # Retornar estadísticas de evaluación

# Entrenar el agente usando Dyna-Q
Q_entrenada = dyna_q()

# Evaluar la política aprendida
evaluar_politica(Q_entrenada)
