import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd

# Wrapper para convertir el entorno en multiobjetivo

class LunarLanderMultiObjetivoWrapper(gym.Wrapper):
    def __init__(self, entorno_original):
        super().__init__(entorno_original)
    
    def step(self, accion):
        observacion, recompensa_original, terminado, truncado, informacion = self.env.step(accion)
        finalizado = terminado or truncado

        if accion != 0:
            penalizacion_por_motor = -0.3 
        else:
            penalizacion_por_motor = 0.0

        vector_recompensas = np.array([recompensa_original, penalizacion_por_motor], dtype=np.float32)

        return observacion, vector_recompensas, terminado, truncado, informacion

    def reset(self, **kwargs):
        observacion_inicial, informacion = self.env.reset(**kwargs)
        return observacion_inicial, informacion


# Red de política (MLP)

class RedPoliticaLunar(nn.Module):
    def __init__(self, dimension_observacion, cantidad_acciones):
        super(RedPoliticaLunar, self).__init__()
        self.capa_oculta1 = nn.Linear(dimension_observacion, 128)
        self.capa_oculta2 = nn.Linear(128, 128)
        self.capa_salida = nn.Linear(128, cantidad_acciones)
        
    def forward(self, entrada_red):
        activacion1 = F.relu(self.capa_oculta1(entrada_red))
        activacion2 = F.relu(self.capa_oculta2(activacion1))
        logits_accion = self.capa_salida(activacion2)
        return F.softmax(logits_accion, dim=-1)


# Ejecutar un episodio

def ejecutar_episodio(entorno, red_politica, dispositivo):
    # Reinicia el entorno y obtiene la observación inicial
    observacion_actual, _ = entorno.reset()

    # Convierte la observación a un tensor flotante en el dispositivo adecuado (CPU o GPU)
    observacion_actual = torch.from_numpy(observacion_actual).float().to(dispositivo)

    # Variable de control para determinar cuándo termina el episodio
    episodio_finalizado = False

    # Lista para guardar los logaritmos de las probabilidades de las acciones tomadas
    log_probabilidades = []

    # Lista para guardar los vectores de recompensas obtenidos en cada paso
    historial_recompensas = []

    # Bucle principal del episodio
    while not episodio_finalizado:
        # Calcula las probabilidades de acción usando la red de política
        probabilidades_acciones = red_politica(observacion_actual)

        # Define una distribución categórica basada en las probabilidades
        distribucion_accion = Categorical(probabilidades_acciones)

        # Muestra (samplea) una acción de la distribución
        accion = distribucion_accion.sample()

        # Calcula el logaritmo de la probabilidad de la acción elegida
        log_prob = distribucion_accion.log_prob(accion)

        # Guarda el logaritmo en la lista
        log_probabilidades.append(log_prob)

        # Convierte el tensor de acción a un entero para ejecutar en el entorno
        accion_ejecutada = accion.item()

        # Ejecuta la acción en el entorno y obtiene la nueva observación y recompensas
        nueva_observacion, vector_recompensas, terminado, truncado, _ = entorno.step(accion_ejecutada)

        # Guarda el vector de recompensas recibido
        historial_recompensas.append(vector_recompensas)

        # Actualiza el estado de finalización del episodio
        episodio_finalizado = terminado or truncado

        # Convierte la nueva observación a tensor para el siguiente paso
        observacion_actual = torch.from_numpy(nueva_observacion).float().to(dispositivo)

    # Convierte la lista de recompensas a un arreglo NumPy
    historial_recompensas = np.array(historial_recompensas, dtype=np.float32)

    # Devuelve los log-probs y las recompensas por paso del episodio
    return log_probabilidades, historial_recompensas



# Retornos descontados

def calcular_retornos_descuento(matriz_recompensas, factor_descuento):
    # Inicializa un arreglo del mismo tamaño que la matriz de recompensas
    # para almacenar los retornos descontados
    retornos_descuento = np.zeros_like(matriz_recompensas, dtype=np.float32)

    # Caso: solo hay una dimensión (una sola recompensa por paso)
    if matriz_recompensas.ndim == 1:
        retorno_acumulado = 0.0
        # Se recorren los pasos del episodio en orden inverso
        for t in reversed(range(len(matriz_recompensas))):
            # Calcula el retorno acumulado descontado hacia atrás
            retorno_acumulado = matriz_recompensas[t] + factor_descuento * retorno_acumulado
            retornos_descuento[t] = retorno_acumulado

    # Caso: múltiples objetivos (matriz 2D)
    else:
        pasos_totales, cantidad_objetivos = matriz_recompensas.shape
        # Se calcula el retorno descontado para cada objetivo por separado
        for indice_objetivo in range(cantidad_objetivos):
            retorno_acumulado = 0.0
            for t in reversed(range(pasos_totales)):
                # Acumula el retorno descontado para el objetivo actual
                recompensa_actual = matriz_recompensas[t, indice_objetivo]
                retorno_acumulado = recompensa_actual + factor_descuento * retorno_acumulado
                retornos_descuento[t, indice_objetivo] = retorno_acumulado

    # Devuelve la matriz de retornos con descuento aplicado
    return retornos_descuento



# Cálculo de dirección de Lara

def calcular_direccion_lara(gradiente_objetivo_1, gradiente_objetivo_2):
    # Pequeño valor para evitar división por cero en la normalización
    epsilon_estabilidad = 1e-8

    # Calcula la norma (magnitud) del primer gradiente
    norma_gradiente_1 = np.linalg.norm(gradiente_objetivo_1) + epsilon_estabilidad

    # Calcula la norma (magnitud) del segundo gradiente
    norma_gradiente_2 = np.linalg.norm(gradiente_objetivo_2) + epsilon_estabilidad

    # Calcula la dirección combinada normalizada (con signo negativo para descenso)
    direccion_combinada = -(
        gradiente_objetivo_1 / norma_gradiente_1 +
        gradiente_objetivo_2 / norma_gradiente_2
    )

    # Devuelve la dirección que será usada para actualizar los parámetros
    return direccion_combinada


def fijar_semilla(valor_semilla=42):
    np.random.seed(valor_semilla)
    torch.manual_seed(valor_semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(valor_semilla)


# Función principal para entrenar una política con el método de Lara
def entrenar_con_lara(seed=42):

    fijar_semilla(seed)
    entorno_original = gym.make("LunarLander-v3")
    entorno_original.reset(seed=seed)

    # Número total de episodios de entrenamiento
    cantidad_episodios = 5000

    # Factor de descuento para las recompensas
    factor_descuento = 0.99

    # Tasa de aprendizaje para actualizar la política
    tasa_aprendizaje = 1e-2

    # Selecciona el dispositivo de cómputo (GPU si está disponible)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crea el entorno base y lo convierte en multiobjetivo
    entorno_original = gym.make("LunarLander-v3")
    entorno = LunarLanderMultiObjetivoWrapper(entorno_original)

    # Dimensión del espacio de observación y número de acciones posibles
    dimension_observacion = entorno_original.observation_space.shape[0]
    cantidad_acciones = entorno_original.action_space.n

    # Inicializa la red de política y la mueve al dispositivo
    red_politica = RedPoliticaLunar(dimension_observacion, cantidad_acciones).to(dispositivo)

    # Lista para almacenar las recompensas totales por episodio
    historial_recompensas_totales = []

    # Bucle principal de entrenamiento por episodio
    for episodio in range(cantidad_episodios):
        # Ejecuta un episodio y obtiene los log-probs y las recompensas por paso
        logaritmos_probabilidades, recompensas_por_paso = ejecutar_episodio(
            entorno, red_politica, dispositivo
        )

        # Número de pasos en el episodio y objetivos
        pasos_episodio = len(recompensas_por_paso)
        cantidad_objetivos = recompensas_por_paso.shape[1]

        # Calcula la recompensa total por objetivo (sin descuento)
        recompensa_total_episodio = np.sum(recompensas_por_paso, axis=0)
        historial_recompensas_totales.append(recompensa_total_episodio)

        # Calcula los retornos con descuento
        retornos_descuento = calcular_retornos_descuento(recompensas_por_paso, factor_descuento)

        # Calcula la pérdida para cada objetivo
        perdidas_por_objetivo = []
        for i_obj in range(cantidad_objetivos):
            retorno_tensor = torch.tensor(retornos_descuento[:, i_obj], dtype=torch.float32).to(dispositivo)
            perdida = sum(-log_prob * retorno for log_prob, retorno in zip(logaritmos_probabilidades, retorno_tensor))
            perdidas_por_objetivo.append(perdida)

        # Calcula los gradientes de cada pérdida respecto a los parámetros de la política
        lista_gradientes = []
        for perdida in perdidas_por_objetivo:
            gradientes_crudos = torch.autograd.grad(perdida, red_politica.parameters(), retain_graph=True)
            vector_gradiente = torch.nn.utils.parameters_to_vector(gradientes_crudos)
            lista_gradientes.append(vector_gradiente.detach().cpu().numpy())

        # Obtiene los gradientes para los dos objetivos
        gradiente_1, gradiente_2 = lista_gradientes[0], lista_gradientes[1]

        # Calcula la dirección de Lara combinando ambos gradientes normalizados
        direccion_actualizacion = calcular_direccion_lara(gradiente_1, gradiente_2)

        # Aplica la actualización de parámetros manualmente
        with torch.no_grad():
            parametros_actuales = torch.nn.utils.parameters_to_vector(red_politica.parameters())
            parametros_actuales += tasa_aprendizaje * torch.tensor(direccion_actualizacion, dtype=parametros_actuales.dtype)
            torch.nn.utils.vector_to_parameters(parametros_actuales, red_politica.parameters())

        # Calcula métricas informativas para el usuario
        promedio_retornos = np.mean(retornos_descuento, axis=0)
        magnitud_direccion = np.linalg.norm(direccion_actualizacion)
                # Calcula métricas informativas cada 100 episodios
        if (episodio + 1) % 100 == 0:
            promedio_retornos = np.mean(retornos_descuento, axis=0)
            magnitud_direccion = np.linalg.norm(direccion_actualizacion)

            # Imprime los valores cada 100 episodios con estilo claro
            print(f"[Episodio {episodio+1:>4}] → Promedio Retorno Obj.1: {promedio_retornos[0]:.2f} | "
                  f"Obj.2: {promedio_retornos[1]:.2f} | Magnitud dirección: {magnitud_direccion:.3f}")


    # Cierra el entorno al finalizar el entrenamiento
    entorno.close()

    # Convierte la lista de recompensas totales a un arreglo NumPy
    historial_recompensas_totales = np.array(historial_recompensas_totales)

    # Gráfica de la evolución de la recompensa total por objetivo
    plt.figure(figsize=(10, 6))
    episodios = np.arange(1, cantidad_episodios + 1)
    for i_obj in range(historial_recompensas_totales.shape[1]):
        plt.plot(episodios, historial_recompensas_totales[:, i_obj], label=f"Objetivo {i_obj+1}")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Total (sin descuento)")
    plt.title("Recompensas por episodio")
    plt.legend()
    plt.grid(True)
    plt.savefig("Descenso.png")
    plt.show()

    # Estadísticas finales (promedio y desviación estándar)
    promedios_finales = np.mean(historial_recompensas_totales, axis=0)
    desviaciones_finales = np.std(historial_recompensas_totales, axis=0)
    resumen_estadistico = {
        "Objetivo": [f"Objetivo {i+1}" for i in range(historial_recompensas_totales.shape[1])],
        "Promedio": promedios_finales,
        "Desviación Estándar": desviaciones_finales
    }
    df_resumen = pd.DataFrame(resumen_estadistico)
    print("\nEstadísticas de la recompensa total por episodio:")
    print(df_resumen)

if __name__ == '__main__':
    entrenar_con_lara(seed=42)
