import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Crear el entorno
entorno = gym.make("CartPole-v1")

# Hiperparámetros
tasa_aprendizaje_politica = 0.01  # Tasa de aprendizaje para la política
tasa_aprendizaje_valor = 0.01  # Tasa de aprendizaje para la función de valor
descuento_gamma = 0.99  # Factor de descuento

dim_estado = entorno.observation_space.shape[0]
dim_accion = entorno.action_space.n

def politica(estado, parametros_politica):
    """Política de selección de acción basada en softmax."""
    preferencias = estado @ parametros_politica  # Preferencias lineales de acción
    exp_preferencias = np.exp(preferencias - np.max(preferencias))  # Ajuste de estabilidad numérica
    return exp_preferencias / np.sum(exp_preferencias)

def funcion_valor(estado, pesos_valor):
    """Aproximación de la función de valor del estado."""
    return np.dot(estado, pesos_valor)

def actor_critico():
    """Entrena un agente Actor-Crítico en el entorno CartPole."""
    pesos_valor = np.zeros(dim_estado)  # Pesos para la función de valor
    parametros_politica = np.random.rand(dim_estado, dim_accion)  # Parámetros de la política
    recompensas_por_episodio = []  # Almacena las recompensas obtenidas por episodio
    
    for episodio in range(1000):  # Número de episodios de entrenamiento
        estado, _ = entorno.reset()  # Inicializar el entorno
        importancia = 1  # Factor de importancia para el gradiente
        terminado = False
        recompensa_total = 0
        
        while not terminado:
            # Calcular la política y muestrear una acción
            probabilidades = politica(estado, parametros_politica)
            accion = np.random.choice(dim_accion, p=probabilidades)
            
            # Ejecutar la acción y observar el siguiente estado y recompensa
            siguiente_estado, recompensa, terminado_env, truncado, _ = entorno.step(accion)
            terminado = terminado_env or truncado
            recompensa_total += recompensa
            
            # Calcular el error de diferencia temporal (TD error)
            delta = recompensa + descuento_gamma * funcion_valor(siguiente_estado, pesos_valor) * (not terminado) - funcion_valor(estado, pesos_valor)
            
            # Actualizar los pesos de la función de valor
            pesos_valor += tasa_aprendizaje_valor * delta * estado
            
            # Actualizar los parámetros de la política
            parametros_politica[:, accion] += tasa_aprendizaje_politica * importancia * delta * estado
            
            # Actualizar el factor de importancia
            importancia *= descuento_gamma
            
            # Moverse al siguiente estado
            estado = siguiente_estado
        
        recompensas_por_episodio.append(recompensa_total)
        if episodio % 100 == 0:
            print(f"Episodio {episodio} completado.")
    
    # Graficar las recompensas obtenidas en cada episodio
    plt.plot(recompensas_por_episodio)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Total")
    plt.title("Progreso del Entrenamiento - Actor-Crítico en CartPole")
    plt.show()
    
    return parametros_politica

def evaluar_politica(parametros_politica, num_ejecuciones=10):
    """Evalúa la política entrenada ejecutándola en el entorno varias veces."""
    recompensas = []
    for _ in range(num_ejecuciones):
        estado, _ = entorno.reset()
        terminado = False
        recompensa_total = 0
        
        while not terminado:
            # Seleccionar la mejor acción según la política aprendida
            probabilidades = politica(estado, parametros_politica)
            accion = np.argmax(probabilidades)
            estado, recompensa, terminado_env, truncado, _ = entorno.step(accion)
            recompensa_total += recompensa
            terminado = terminado_env or truncado
        
        recompensas.append(recompensa_total)
    
    # Calcular la media y desviación estándar de las recompensas obtenidas
    recompensa_media = np.mean(recompensas)
    desviacion_recompensa = np.std(recompensas)
    print(f"Evaluación en {num_ejecuciones} ejecuciones: Recompensa Media = {recompensa_media}, Desviación Estándar = {desviacion_recompensa}")
    return recompensa_media, desviacion_recompensa

# Entrenar el agente
parametros_politica_entrenados = actor_critico()

# Evaluar la política aprendida
evaluar_politica(parametros_politica_entrenados)
