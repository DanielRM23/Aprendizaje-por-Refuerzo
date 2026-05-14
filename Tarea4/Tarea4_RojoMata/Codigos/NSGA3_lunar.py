# ----------------------------------------------------------
# NSGA-III adaptado a MO-Lunar-Lander: evaluación y selección ambiental
# ----------------------------------------------------------

# Importa el entorno multiobjetivo de Gymnasium
import mo_gymnasium as mo_gym

# Importaciones para operaciones numéricas, manejo de distancias y estadísticas
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist  # Para calcular distancias (como coseno)
from scipy.linalg import LinAlgError      # Para manejar errores en operaciones matriciales
from itertools import combinations        # Para generar combinaciones (puntos de referencia)
import matplotlib.pyplot as plt           # Para visualización de resultados


# ----------------------------------------------------------
# 1. Crear entorno MO-Lunar-Lander sin scalarizar
# ----------------------------------------------------------

# Se crea una instancia del entorno multiobjetivo Lunar Lander
# Este entorno devuelve recompensas vectoriales sin combinarlas (no scalarizado)
entorno = mo_gym.make('mo-lunar-lander-v3')


# ----------------------------------------------------------
# 2. Conversión entre vector plano y política lineal
# ----------------------------------------------------------

def vector_a_politica(vector_pesos, dim_obs=8, dim_act=4):
    """
    Convierte un vector plano de pesos en una matriz que representa una política lineal.

    Parámetros:
        vector_pesos : np.ndarray
            Vector plano de pesos con tamaño (dim_act * dim_obs).
        dim_obs : int
            Número de observaciones (dimensión del estado).
        dim_act : int
            Número de acciones posibles.

    Retorna:
        np.ndarray
            Matriz de pesos con forma (dim_act, dim_obs).
    """
    return vector_pesos.reshape((dim_act, dim_obs))

# ----------------------------------------------------------
# 3. Aplicar la política a una observación y devolver acción
# ----------------------------------------------------------

def seleccionar_accion(politica, obs):
    """
    Selecciona una acción dada una observación y una política lineal.

    Parámetros:
        politica : np.ndarray
            Matriz de pesos de la política (dim_act, dim_obs).
        obs : np.ndarray
            Vector de observación del entorno.

    Retorna:
        int
            Índice de la acción seleccionada (la que tiene mayor valor de activación).
    """
    logits = politica @ obs  # Producto entre pesos y observación (una fila por acción)
    return np.argmax(logits)  # Se selecciona la acción con mayor puntuación (mayor logit)


# ----------------------------------------------------------
# 4. Evaluar una política en el entorno MO-Lunar-Lander
# ----------------------------------------------------------

def evaluar_politica(politica, entorno, episodios=5, semilla=None, retornar_todas=False):
    """
    Evalúa una política lineal en el entorno multiobjetivo MO-Lunar-Lander.

    Parámetros:
        politica : np.ndarray
            Matriz de pesos (forma: acciones x observaciones).
        entorno : gym.Env
            Entorno multiobjetivo.
        episodios : int
            Número de episodios a ejecutar para promediar resultados.
        semilla : int o None
            Semilla para reproducibilidad (opcional).
        retornar_todas : bool
            Si es True, retorna todas las recompensas por episodio. Si es False, retorna el promedio.

    Retorna:
        np.ndarray
            Recompensa media por objetivo (o todas si retornar_todas=True).
    """

    if semilla is not None:
        np.random.seed(semilla)  # Fijar semilla si se solicita

    recompensas = []  # Lista para guardar vectores de recompensa por episodio

    for _ in range(episodios):
        obs, _ = entorno.reset()  # Reinicia el entorno
        total = np.zeros(4, dtype=np.float32)  # Vector acumulador de recompensa por objetivo
        terminado, truncado = False, False

        # Ejecuta un episodio completo
        while not (terminado or truncado):
            accion = seleccionar_accion(politica, obs)  # Selecciona acción usando la política
            obs, r_vec, terminado, truncado, _ = entorno.step(accion)  # Paso en el entorno
            total += r_vec  # Suma recompensa vectorial

        recompensas.append(total)  # Guarda resultado del episodio

    recompensas = np.array(recompensas)

    # Retorna todas las recompensas o solo el promedio, según se indique
    return recompensas if retornar_todas else np.mean(recompensas, axis=0)


# ----------------------------------------------------------
# 5. Inicialización de población y evaluación de objetivos
# ----------------------------------------------------------

def inicializar_poblacion(tamano, dimensiones):
    """
    Inicializa una población de individuos con valores entre -1 y 1.

    Parámetros:
        tamano : int
            Número de individuos.
        dimensiones : int
            Número de variables por individuo (dim_act × dim_obs).

    Retorna:
        np.ndarray
            Matriz de individuos (forma: tamano × dimensiones).
    """
    return np.random.uniform(-1, 1, size=(tamano, dimensiones))


def evaluar_poblacion(poblacion, entorno):
    """
    Evalúa toda la población en el entorno, convirtiendo cada vector plano en una política.

    Parámetros:
        poblacion : np.ndarray
            Población de vectores planos (forma: N × (dim_act × dim_obs)).
        entorno : gym.Env
            Entorno multiobjetivo.

    Retorna:
        np.ndarray
            Matriz de recompensas (forma: N × número de objetivos).
    """
    recompensas = []

    for individuo in poblacion:
        politica = vector_a_politica(individuo)  # Convierte vector plano en política
        recompensa = evaluar_politica(politica, entorno)  # Evalúa política en el entorno
        recompensas.append(recompensa)

    return np.array(recompensas)


# ----------------------------------------------------------
# 6. Ordenamiento no dominado (Fast Non-Dominated Sorting)
# ----------------------------------------------------------

def ordenar_no_dominadas(matriz_objetivos):
    """
    Implementa el ordenamiento rápido no dominado para clasificación en frentes de Pareto.

    Parámetros:
        matriz_objetivos : np.ndarray
            Matriz de tamaño (N × M) donde N es el número de individuos y
            M es el número de objetivos.

    Retorna:
        frentes : dict
            Diccionario con claves = número de frente (0, 1, ...) y valores = listas de índices.
        rango : np.ndarray
            Vector con el número de frente asignado a cada individuo.
    """

    num_individuos, num_objetivos = matriz_objetivos.shape

    dominado_por = np.zeros(num_individuos, dtype=int)  # Cuántos individuos dominan a cada uno
    domina_a = [[] for _ in range(num_individuos)]       # A quién domina cada individuo

    frentes = {0: []}  # Diccionario con frentes (nivel 0 es el mejor)
    rango = np.zeros(num_individuos, dtype=int)  # Frente asignado a cada individuo

    # Comparar cada par de individuos para establecer relaciones de dominancia
    for i in range(num_individuos):
        for j in range(num_individuos):
            if i != j:
                mejor, igual, peor = 0, 0, 0
                for k in range(num_objetivos):
                    if matriz_objetivos[i, k] < matriz_objetivos[j, k]:
                        mejor += 1
                    elif matriz_objetivos[i, k] == matriz_objetivos[j, k]:
                        igual += 1
                    else:
                        peor += 1

                # Si i es dominado por j
                if mejor == 0 and igual != num_objetivos:
                    dominado_por[i] += 1
                # Si i domina a j
                elif peor == 0 and igual != num_objetivos:
                    domina_a[i].append(j)

        # Si nadie domina a i, pertenece al primer frente (frente 0)
        if dominado_por[i] == 0:
            frentes[0].append(i)
            rango[i] = 0

    # Construcción de frentes subsiguientes
    i = 0
    while frentes[i]:
        siguiente_frente = []
        for p in frentes[i]:            # Por cada individuo del frente actual
            for q in domina_a[p]:       # Por cada individuo dominado por p
                dominado_por[q] -= 1    # q tiene un dominador menos
                if dominado_por[q] == 0:  # Si ya nadie lo domina, pasa al siguiente frente
                    rango[q] = i + 1
                    siguiente_frente.append(q)
        i += 1
        frentes[i] = siguiente_frente  # Se agrega el nuevo frente

    frentes.pop(i)  # El último frente estará vacío, se elimina

    return frentes, rango


def factorial(n):
    """
    Calcula el factorial de un número de forma recursiva.
    """
    return 1 if n <= 1 else n * factorial(n - 1)


def combinacion(n, m):
    """
    Calcula el coeficiente binomial C(n, m), también conocido como "n sobre m".

    Retorna:
        int : Número de combinaciones posibles de m elementos tomados de un conjunto de n.
    """
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    return factorial(n) // (factorial(m) * factorial(n - m))


def generar_puntos_referencia(tamano_poblacion, num_obj):
    """
    Genera puntos de referencia uniformemente distribuidos en un espacio de múltiples objetivos,
    usando el método de Das y Dennis (1998), ideal para NSGA-III.

    Parámetros:
        tamano_poblacion : int
            Número máximo de puntos a generar (generalmente igual al tamaño de la población).
        num_obj : int
            Número de objetivos.

    Retorna:
        np.ndarray : Matriz de puntos de referencia con forma (N, num_obj).
    """

    h1 = 0  # Número de divisiones en el primer conjunto de puntos

    # Encuentra el mayor h1 tal que el número de combinaciones no exceda el tamaño de población
    while combinacion(h1 + num_obj, num_obj - 1) <= tamano_poblacion:
        h1 += 1

    # Genera combinaciones de índices que se convertirán en puntos
    puntos = np.array(list(combinations(np.arange(1, h1 + num_obj), num_obj - 1))) - np.arange(num_obj - 1) - 1

    # Calcula coordenadas reales normalizadas de los puntos
    puntos = (
        np.concatenate((puntos, np.zeros((puntos.shape[0], 1)) + h1), axis=1) -
        np.concatenate((np.zeros((puntos.shape[0], 1)), puntos), axis=1)
    ) / h1

    # ---------------------------------------------------------------
    # Si aún hay espacio, se generan puntos intermedios adicionales
    # ---------------------------------------------------------------
    if h1 < num_obj:
        h2 = 0
        while combinacion(h1 + num_obj - 1, num_obj - 1) + combinacion(h2 + num_obj, num_obj - 1) <= tamano_poblacion:
            h2 += 1

        if h2 > 0:
            # Genera puntos con una segunda capa (más finos)
            puntos_extra = np.array(list(combinations(np.arange(1, h2 + num_obj), num_obj - 1))) - np.arange(num_obj - 1) - 1
            puntos_extra = (
                np.concatenate((puntos_extra, np.zeros((puntos_extra.shape[0], 1)) + h2), axis=1) -
                np.concatenate((np.zeros((puntos_extra.shape[0], 1)), puntos_extra), axis=1)
            ) / h2

            # Desplaza los puntos intermedios a la mitad entre direcciones originales
            puntos_extra = puntos_extra / 2 + 1 / (2 * num_obj)

            # Junta los puntos originales y los intermedios
            puntos = np.concatenate((puntos, puntos_extra), axis=0)

    return puntos


# ----------------------------------------------------------
# 7. Evaluar políticas bajo recompensas escalarizadas
# ----------------------------------------------------------

def evaluar_politicas_scalarizadas(poblacion, recompensas, pesos_list, entorno_original, episodios=5):
    """
    Evalúa un subconjunto de políticas del frente de Pareto usando recompensas escalarizadas
    (combinaciones lineales de múltiples objetivos), con distintas configuraciones de pesos.

    Parámetros:
        poblacion : np.ndarray
            Conjunto de individuos (vectores planos) del frente de Pareto.
        recompensas : np.ndarray
            Recompensas vectoriales asociadas a la población (no se usa directamente aquí).
        pesos_list : list[np.ndarray]
            Lista de vectores de pesos (uno por configuración de scalarización).
        entorno_original : gym.Env
            Entorno original MO-Gym sin scalarizar (se usará env wrapper para scalarizar).
        episodios : int
            Número de episodios a ejecutar por política para estimar recompensa escalar.
    """
    
    print("\nEvaluación con diferentes configuraciones de pesos:")

    for i, w in enumerate(pesos_list):
        # Crea un entorno scalarizado con los pesos dados (linear scalarization)
        entorno_scalar = mo_gym.wrappers.LinearReward(entorno_original, weight=w)

        print(f"\n👉 Configuración de pesos {i+1}: {w}")

        # Se seleccionan las 3 primeras políticas de la población como representativas
        for j in range(3):
            politica = vector_a_politica(poblacion[j])  # Convierte vector plano a matriz de política
            recompensa_scalar = 0

            # Ejecuta varios episodios para obtener promedio
            for _ in range(episodios):
                obs, _ = entorno_scalar.reset()  # Reinicia el entorno
                total = 0
                done = False

                # Ejecuta episodio completo
                while not done:
                    accion = seleccionar_accion(politica, obs)  # Acción según política lineal
                    obs, reward, terminated, truncated, _ = entorno_scalar.step(accion)
                    done = terminated or truncated
                    total += reward  # Acumula recompensa escalar

                recompensa_scalar += total

            # Promedio de recompensa escalarizada sobre los episodios
            recompensa_prom = recompensa_scalar / episodios
            print(f"  Política {j+1}: Recompensa escalar promedio = {recompensa_prom:.2f}")


# ----------------------------------------------------------
# 8. Normalización de recompensas respecto al frente 0
# ----------------------------------------------------------

def normalizar_recompensas(recompensas, indices_frente0):
    """
    Normaliza las recompensas multiobjetivo entre 0 y 1, usando como referencia
    el punto ideal (mínimo global) y el punto nadir (máximo en el frente 0).

    Parámetros:
        recompensas : np.ndarray
            Matriz de recompensas vectoriales de la población (N × M).
        indices_frente0 : list[int]
            Índices de los individuos que pertenecen al frente de Pareto (frente 0).

    Retorna:
        np.ndarray
            Matriz normalizada de recompensas, donde cada objetivo está en [0, 1].
    """

    # Punto ideal: el valor mínimo observado para cada objetivo
    ideal = recompensas.min(axis=0)

    # Punto nadir: máximo observado en el frente 0 si se proporciona,
    # o en toda la población si no hay frente 0
    if len(indices_frente0) == 0:
        nadir = recompensas.max(axis=0)
    else:
        nadir = recompensas[indices_frente0].max(axis=0)

    # Normaliza cada recompensa individualmente con respecto a los extremos
    # Se suma 1e-12 para evitar división por cero
    return (recompensas - ideal) / (nadir - ideal + 1e-12)


# ----------------------------------------------------------
# 9. Selección ambiental (NSGA-III)
# ----------------------------------------------------------

def seleccion_ambiental(poblacion, recompensas, referencias, tamano_poblacion):
    """
    Realiza la selección ambiental de NSGA-III para reducir una población extendida
    a un tamaño deseado, basándose en dominancia y asignación a nichos.

    Parámetros:
        poblacion : np.ndarray
            Matriz de individuos (N × D).
        recompensas : np.ndarray
            Matriz de recompensas vectoriales (N × M).
        referencias : np.ndarray
            Direcciones de referencia (nichos).
        tamano_poblacion : int
            Tamaño objetivo de la población.

    Retorna:
        nueva_poblacion : np.ndarray
            Conjunto de individuos seleccionados.
        nuevas_recompensas : np.ndarray
            Recompensas de los seleccionados.
        nuevos_rangos : np.ndarray
            Rangos de los individuos seleccionados.
    """

    # Paso 1: Ordenamiento no dominado
    frentes, rangos = ordenar_no_dominadas(recompensas)
    num_obj = recompensas.shape[1]

    # Paso 2: Seleccionar tantos frentes como quepan completamente
    seleccionados = np.full(poblacion.shape[0], False)
    frente_actual = 0
    while np.sum(seleccionados) + len(frentes[frente_actual]) <= tamano_poblacion:
        seleccionados[frentes[frente_actual]] = True
        frente_actual += 1

    # Número de individuos faltantes para completar la población
    faltantes = tamano_poblacion - np.sum(seleccionados)

    # Paso 3: Normalizar recompensas usando punto ideal y nadir del conjunto combinado
    recompensas_normalizadas = normalizar_recompensas(
        np.vstack([recompensas[seleccionados], recompensas[frentes[frente_actual]]]),
        list(range(np.sum(seleccionados)))
    )

    # Paso 4: Calcular distancia perpendicular a cada dirección de referencia
    distancias = np.sqrt((recompensas_normalizadas**2).sum(axis=1)).reshape(-1, 1)  # ||F||
    proyecciones = 1 - cdist(recompensas_normalizadas, referencias, metric="cosine")  # cos(θ)
    distancias_perpendiculares = distancias * np.sqrt(1 - proyecciones**2)  # ||F||·sin(θ)

    # Paso 5: Asociar cada solución al nicho más cercano (menor distancia perpendicular)
    asociaciones = np.argmin(distancias_perpendiculares, axis=1)

    # Paso 6: Contar soluciones ya seleccionadas por nicho
    conteos = Counter(asociaciones[:np.sum(seleccionados)])
    conteo_por_nicho = np.zeros(len(referencias))
    for k, v in conteos.items():
        conteo_por_nicho[k] = v

    # Paso 7: Selección nichada desde el último frente parcial
    seleccion_final = np.full(len(frentes[frente_actual]), False)  # Marca individuos ya elegidos
    referencias_activas = np.full(len(referencias), True)          # Direcciones que aún tienen candidatos
    seleccionados_extra = []  # Índices globales de los seleccionados desde el frente actual

    while np.sum(seleccion_final) < faltantes:
        # Direcciones de referencia aún activas
        disponibles = np.where(referencias_activas)[0]

        # Buscar nichos menos poblados entre los disponibles
        min_nicho = np.min(conteo_por_nicho[disponibles])
        candidatos_nicho = disponibles[conteo_por_nicho[disponibles] == min_nicho]
        nicho_seleccionado = np.random.choice(candidatos_nicho)

        # Candidatos en el frente actual que están asociados al nicho seleccionado
        candidatos_sol = np.where(
            (asociaciones[np.sum(seleccionados):] == nicho_seleccionado) & (~seleccion_final)
        )[0]

        if len(candidatos_sol) > 0:
            # Si el nicho está vacío, selecciona el más cercano; si no, uno aleatorio
            if conteo_por_nicho[nicho_seleccionado] == 0:
                mejor = candidatos_sol[np.argmin(
                    distancias_perpendiculares[np.sum(seleccionados) + candidatos_sol, nicho_seleccionado])]
            else:
                mejor = np.random.choice(candidatos_sol)

            seleccion_final[mejor] = True
            seleccionados_extra.append(frentes[frente_actual][mejor])
            conteo_por_nicho[nicho_seleccionado] += 1
        else:
            # Si no hay candidatos para ese nicho, se desactiva
            referencias_activas[nicho_seleccionado] = False

    # Paso 8: Combinar seleccionados completos y los del último frente
    indices_finales = list(np.where(seleccionados)[0]) + seleccionados_extra

    # Retornar población, objetivos y rangos seleccionados
    nueva_poblacion = poblacion[indices_finales]
    nuevas_recompensas = recompensas[indices_finales]
    nuevos_rangos = np.array(rangos)[indices_finales]

    return nueva_poblacion, nuevas_recompensas, nuevos_rangos


# ----------------------------------------------------------
# 10. Selección por torneo binario
# ----------------------------------------------------------

def seleccion_torneo(poblacion, prob_cruce, rangos, k=2):
    """
    Realiza selección por torneo binario para construir el conjunto de apareamiento.

    Parámetros:
        poblacion : np.ndarray
            Matriz de individuos (N × D), donde N es el número de individuos y D las variables.
        prob_cruce : float
            Proporción de la población que será seleccionada para cruce.
        rangos : np.ndarray
            Vector que indica el rango (nivel de dominancia) de cada individuo.
        k : int
            Tamaño del torneo (por defecto 2, torneo binario).

    Retorna:
        conjunto_cruza : np.ndarray
            Conjunto de individuos seleccionados para cruce (M × D).
    """

    tamano_poblacion, num_variables = poblacion.shape

    # Calcula cuántos individuos deben seleccionarse para cruce
    num_cruzas = int(tamano_poblacion * prob_cruce)

    # Se asegura que el número de cruzas sea par (requerido para cruce por pares)
    if num_cruzas % 2 != 0:
        num_cruzas += 1

    # Inicializa el conjunto donde se almacenarán los individuos seleccionados
    conjunto_cruza = np.zeros((num_cruzas, num_variables))

    for i in range(num_cruzas):
        # Selecciona k individuos al azar (sin reemplazo) para el torneo
        ind1, ind2 = np.random.choice(tamano_poblacion, k, replace=False)

        # Gana el que tenga menor rango (más dominante)
        if rangos[ind1] <= rangos[ind2]:
            conjunto_cruza[i] = poblacion[ind1]
        else:
            conjunto_cruza[i] = poblacion[ind2]

    return conjunto_cruza


# ----------------------------------------------------------
# 11. Operador de cruce simulado binario (SBX)
# ----------------------------------------------------------

def operador_cruce(conjunto_cruza, limites_inferiores, limites_superiores, prob_cruce, eta_distribucion):
    """
    Aplica el operador de cruce SBX (Simulated Binary Crossover) sobre un conjunto de padres.

    Parámetros:
        conjunto_cruza : np.ndarray
            Conjunto de individuos seleccionados (N × D).
        limites_inferiores : np.ndarray
            Límite inferior por variable (vector de D elementos).
        limites_superiores : np.ndarray
            Límite superior por variable (vector de D elementos).
        prob_cruce : float
            Probabilidad de que se aplique el cruce por pareja.
        eta_distribucion : float
            Parámetro de distribución SBX (mayores valores = hijos más similares a padres).

    Retorna:
        hijos : np.ndarray
            Nuevos individuos generados tras el cruce (N × D).
    """

    num_hijos, num_variables = conjunto_cruza.shape
    mitad = int(num_hijos / 2)  # Se asume que num_hijos es par

    # División en parejas
    padres1 = conjunto_cruza[:mitad]
    padres2 = conjunto_cruza[mitad:]

    # Inicializa matriz beta, que determina la contribución relativa de cada padre
    beta = np.zeros((mitad, num_variables))

    # Genera números aleatorios para calcular beta según la fórmula de SBX
    mu = np.random.random((mitad, num_variables))

    # Cálculo de beta para cada variable según SBX
    condicion1 = mu <= 0.5
    condicion2 = ~condicion1
    beta[condicion1] = (2 * mu[condicion1]) ** (1 / (eta_distribucion + 1))
    beta[condicion2] = (2 - 2 * mu[condicion2]) ** (-1 / (eta_distribucion + 1))

    # Aleatoriza el signo de beta (permite explorar el espacio entre los padres)
    beta = beta * (-1) ** np.random.randint(0, 2, (mitad, num_variables))

    # Con 50% de probabilidad, fuerza beta = 1 → hijo igual a promedio
    beta[np.random.random((mitad, num_variables)) < 0.5] = 1

    # Si no se cumple la probabilidad de cruce, se fuerza beta = 1 (sin cruce)
    beta[np.tile(np.random.random((mitad, 1)) > prob_cruce, (1, num_variables))] = 1

    # Cálculo de hijos (hijos1 y hijos2 son simétricos respecto al promedio de los padres)
    hijos1 = (padres1 + padres2) / 2 + beta * (padres1 - padres2) / 2
    hijos2 = (padres1 + padres2) / 2 - beta * (padres1 - padres2) / 2

    # Se combinan ambos hijos en una sola matriz
    hijos = np.concatenate((hijos1, hijos2), axis=0)

    # Se asegura que los hijos respeten los límites del dominio
    hijos = np.minimum(hijos, np.tile(limites_superiores, (num_hijos, 1)))
    hijos = np.maximum(hijos, np.tile(limites_inferiores, (num_hijos, 1)))

    return hijos


# ----------------------------------------------------------
# 12. Operador de mutación polinomial
# ----------------------------------------------------------

def operador_mutacion(poblacion, limites_inferiores, limites_superiores, prob_mutacion, eta_mutacion):
    """
    Aplica el operador de mutación polinómica a una población de individuos reales.

    Parámetros:
        poblacion : np.ndarray
            Matriz de individuos (N × D).
        limites_inferiores : np.ndarray
            Límite inferior por variable.
        limites_superiores : np.ndarray
            Límite superior por variable.
        prob_mutacion : float
            Probabilidad total de aplicar mutación a un individuo.
        eta_mutacion : float
            Parámetro de control de la distribución de la mutación.

    Retorna:
        np.ndarray
            Población mutada (respetando los límites).
    """

    tamano_poblacion, num_variables = poblacion.shape

    # Expande los límites para que coincidan con la forma de la población
    limites_inferiores = np.tile(limites_inferiores, (tamano_poblacion, 1))
    limites_superiores = np.tile(limites_superiores, (tamano_poblacion, 1))

    # Genera máscara booleana que indica en qué genes se aplica la mutación
    mascara_mutacion = np.random.random((tamano_poblacion, num_variables)) < prob_mutacion / num_variables

    # Valores aleatorios para determinar tipo de mutación (menor o mayor a 0.5)
    valores_aleatorios = np.random.random((tamano_poblacion, num_variables))

    # Cálculo de distancia relativa a los límites (normalizada)
    delta_inferior = (poblacion - limites_inferiores) / (limites_superiores - limites_inferiores)
    delta_superior = (limites_superiores - poblacion) / (limites_superiores - limites_inferiores)

    # --------------------------
    # Mutaciones con r <= 0.5
    # --------------------------
    condicion_baja = np.logical_and(mascara_mutacion, valores_aleatorios <= 0.5)
    poblacion[condicion_baja] += (limites_superiores[condicion_baja] - limites_inferiores[condicion_baja]) * (
        (2 * valores_aleatorios[condicion_baja] +
         (1 - 2 * valores_aleatorios[condicion_baja]) *
         (1 - delta_inferior[condicion_baja]) ** (eta_mutacion + 1)
        ) ** (1 / (eta_mutacion + 1)) - 1
    )

    # --------------------------
    # Mutaciones con r > 0.5
    # --------------------------
    condicion_alta = np.logical_and(mascara_mutacion, valores_aleatorios > 0.5)
    poblacion[condicion_alta] += (limites_superiores[condicion_alta] - limites_inferiores[condicion_alta]) * (
        1 - (2 * (1 - valores_aleatorios[condicion_alta]) +
             2 * (valores_aleatorios[condicion_alta] - 0.5) *
             (1 - delta_superior[condicion_alta]) ** (eta_mutacion + 1)
            ) ** (1 / (eta_mutacion + 1))
    )

    # Forzar que la población mutada esté dentro de los límites
    poblacion = np.minimum(poblacion, limites_superiores)
    poblacion = np.maximum(poblacion, limites_inferiores)

    return poblacion


# ----------------------------------------------------------
# 13. Entrenamiento principal de NSGA-III
# ----------------------------------------------------------

def entrenar_nsga3_mo_lunar(
    tamano_poblacion=100,
    num_generaciones=1000,
    prob_cruce=1.0,
    prob_mutacion=1.0,
    eta_cruce=30,
    eta_mutacion=20
):
    """
    Ejecuta NSGA-III desde cero sobre el entorno MO-Lunar-Lander con políticas lineales.

    Parámetros:
        tamano_poblacion : int
            Número de individuos por generación.
        num_generaciones : int
            Número total de generaciones a ejecutar.
        prob_cruce : float
            Probabilidad de aplicar cruce SBX.
        prob_mutacion : float
            Probabilidad de aplicar mutación polinómica.
        eta_cruce : float
            Parámetro de control del cruce SBX (más alto = hijos parecidos a padres).
        eta_mutacion : float
            Parámetro de control de la mutación polinómica.
    """

    # Tamaño del vector de políticas lineales: 4 acciones × 8 observaciones
    dimensiones = 32
    num_objetivos = 4

    # Límites de búsqueda para cada variable (peso de política)
    limites_inf = np.full(dimensiones, -1.0)
    limites_sup = np.full(dimensiones, 1.0)

    # Inicialización de población y evaluación inicial
    poblacion = inicializar_poblacion(tamano_poblacion, dimensiones)
    recompensas = evaluar_poblacion(poblacion, entorno)

    # Generación de direcciones de referencia para NSGA-III
    referencias = generar_puntos_referencia(tamano_poblacion, num_objetivos)

    recompensas_generacion = []  # Para graficar evolución por objetivo

    # Bucle principal de generaciones
    for gen in range(num_generaciones):
        if (gen + 1) % 10 == 0:
            print(f"Generación {gen + 1}/{num_generaciones}")

        # Ordenamiento por dominancia
        frentes, rangos = ordenar_no_dominadas(recompensas)

        # Almacenar promedio de recompensas por objetivo
        recompensas_generacion.append(np.mean(recompensas, axis=0))

        # Operadores evolutivos: selección, cruce y mutación
        padres = seleccion_torneo(poblacion, prob_cruce, rangos)
        hijos = operador_cruce(padres, limites_inf, limites_sup, prob_cruce, eta_cruce)
        hijos = operador_mutacion(hijos, limites_inf, limites_sup, prob_mutacion, eta_mutacion)

        # Evaluación de la nueva población
        recompensas_hijos = evaluar_poblacion(hijos, entorno)

        # Combinar padres e hijos y realizar selección ambiental
        poblacion_combinada = np.vstack([poblacion, hijos])
        recompensas_combinadas = np.vstack([recompensas, recompensas_hijos])
        poblacion, recompensas, _ = seleccion_ambiental(
            poblacion_combinada, recompensas_combinadas, referencias, tamano_poblacion
        )

    # --------------------------------------------
    # Visualización del frente de Pareto aproximado
    # --------------------------------------------
    frente_pareto = recompensas

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(frente_pareto[:, 0], frente_pareto[:, 1], frente_pareto[:, 2], color='red')
    ax.set_xlabel('Objetivo 1')
    ax.set_ylabel('Objetivo 2')
    ax.set_zlabel('Objetivo 3')
    ax.set_title('Frente de Pareto aproximado - MO-Lunar-Lander')
    plt.tight_layout()
    plt.show()

    # --------------------------------------------
    # Evolución de recompensas promedio por objetivo
    # --------------------------------------------
    recompensas_generacion = np.array(recompensas_generacion)
    for i in range(recompensas_generacion.shape[1]):
        plt.plot(recompensas_generacion[:, i], label=f'Objetivo {i+1}')
    plt.xlabel('Generación')
    plt.ylabel('Recompensa promedio')
    plt.title('Evolución de recompensas por generación')
    plt.legend()
    plt.grid(True)
    plt.show()


# Ejecutar directamente si se corre el archivo
if __name__ == "__main__":
    entrenar_nsga3_mo_lunar()
