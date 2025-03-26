# Importamos las librerías necesarias
from minatar import Environment  # Importamos el entorno MinAtar
import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para graficar resultados
import random  # Para selección aleatoria en la planificación

# Parámetros del entorno
ejuego = "breakout"  # Seleccionamos el juego "breakout" en MinAtar
entorno = Environment(ejuego)  # Inicializamos el entorno
num_acciones = entorno.num_actions()  # Obtenemos el número de acciones posibles
forma_estado = entorno.state_shape()  # Obtenemos la forma del estado
dim_estado = np.prod(forma_estado)  # Calculamos la dimensión total del estado

# Función para preprocesar el estado
def preprocesar_estado(estado):
    return tuple((estado.flatten() * 10).astype(int))  # Aplanamos, escalamos y discretizamos el estado

# Inicializamos la tabla Q como un diccionario para manejar estados discretizados
Q = {}

# Hiperparámetros de Dyna-Q
alpha = 0.01  # Tasa de aprendizaje (cuánto ajustamos los valores Q)
gamma = 0.99  # Factor de descuento (importancia de recompensas futuras)
epsilon_inicial = 0.4  # Probabilidad inicial de exploración (para acciones aleatorias)
epsilon_min = 0.05  # Valor mínimo de epsilon (para asegurar algo de exploración)
num_episodios = 5000  # Número total de episodios de entrenamiento
num_planificaciones = 10  # Número de iteraciones de planificación por paso

# Diccionario que actúa como el modelo del entorno (almacena transiciones)
modelo = {}

# Función para seleccionar una acción usando una estrategia epsilon-greedy
def seleccionar_accion(estado, epsilon):
    if np.random.rand() < epsilon:  # Exploración: seleccionamos una acción aleatoria
        return np.random.choice(num_acciones)
    
    if estado not in Q:  # Si el estado es nuevo, inicializamos sus valores Q
        Q[estado] = np.zeros(num_acciones)
    
    return np.argmax(Q[estado])  # Explotación: elegimos la acción con mayor valor Q

# Función para actualizar la tabla Q
def actualizar_Q(estado, accion, recompensa, siguiente_estado, terminado):
    if estado not in Q:  # Si el estado no está en Q, lo inicializamos
        Q[estado] = np.zeros(num_acciones)
    if siguiente_estado not in Q:  # También inicializamos el siguiente estado si es nuevo
        Q[siguiente_estado] = np.zeros(num_acciones)
    
    # Calculamos el valor objetivo usando la ecuación de actualización Q-learning
    objetivo = recompensa + (gamma * np.max(Q[siguiente_estado]) if not terminado else 0)
    
    # Actualizamos el valor Q usando la ecuación de aprendizaje
    Q[estado][accion] += alpha * (objetivo - Q[estado][accion])

# Función para actualizar el modelo del entorno con la transición observada
def actualizar_modelo(estado, accion, recompensa, siguiente_estado, terminado):
    modelo[(estado, accion)] = (recompensa, siguiente_estado, terminado)

# Función para realizar planificación en Dyna-Q
def planificacion():
    for _ in range(num_planificaciones):  # Iteramos sobre varias simulaciones de planificación
        if not modelo:  # Si el modelo está vacío, no hacemos nada
            return
        (s, a), (r, s_prime, done) = random.choice(list(modelo.items()))  # Seleccionamos una transición al azar
        actualizar_Q(s, a, r, s_prime, done)  # Aplicamos la actualización de Q-learning

# Lista para almacenar las recompensas por episodio
recompensas_por_episodio = []

# Ciclo principal de entrenamiento
for episodio in range(num_episodios):
    entorno.reset()  # Reiniciamos el entorno al inicio de cada episodio
    estado = preprocesar_estado(entorno.state())  # Obtenemos y preprocesamos el estado inicial
    
    # Calculamos epsilon usando una estrategia de disminución lineal
    epsilon = max(epsilon_inicial - episodio / (num_episodios / 2), epsilon_min)
    
    terminado = False  # Indicador de si el episodio ha terminado
    recompensa_total = 0  # Almacenamos la recompensa total del episodio
    
    while not terminado:  # Iteramos hasta que el episodio termine
        accion = seleccionar_accion(estado, epsilon)  # Seleccionamos una acción usando epsilon-greedy
        recompensa, terminado = entorno.act(accion)  # Ejecutamos la acción en el entorno
        
        recompensa = max(recompensa, -10)  # Limitamos la recompensa mínima a -10
        
        siguiente_estado = preprocesar_estado(entorno.state())  # Obtenemos y preprocesamos el nuevo estado
        recompensa_total += recompensa  # Acumulamos la recompensa del episodio
        
        actualizar_Q(estado, accion, recompensa, siguiente_estado, terminado)  # Actualizamos Q
        actualizar_modelo(estado, accion, recompensa, siguiente_estado, terminado)  # Actualizamos el modelo
        planificacion()  # Realizamos planificación con el modelo
        
        estado = siguiente_estado  # Avanzamos al siguiente estado
    
    recompensas_por_episodio.append(recompensa_total)  # Almacenamos la recompensa total del episodio
    
    # Imprimimos información cada 100 episodios
    if (episodio + 1) % 100 == 0:
        print(f"Episodio {episodio+1}/{num_episodios} - Recompensa Total: {recompensa_total}")

# Graficamos la recompensa obtenida por episodio
plt.plot(range(num_episodios), recompensas_por_episodio)
plt.xlabel("Episodio")  # Etiqueta del eje X
plt.ylabel("Recompensa acumulada")  # Etiqueta del eje Y
plt.title("Recompensa obtenida por episodio con Dyna-Q")  # Título del gráfico
plt.show()  # Mostramos la gráfica

# Calculamos estadísticas sobre las recompensas obtenidas
recompensa_media = np.mean(recompensas_por_episodio)  # Calculamos la media
recompensa_varianza = np.var(recompensas_por_episodio)  # Calculamos la varianza

# Mostramos las estadísticas calculadas
print(f"Recompensa media: {recompensa_media:.2f}")
print(f"Varianza de la recompensa: {recompensa_varianza:.2f}")
