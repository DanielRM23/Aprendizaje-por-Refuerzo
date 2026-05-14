# Librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from pymoo.problems.many.wfg import WFG1
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from scipy.spatial.distance import cdist
from collections import Counter
from scipy.linalg import LinAlgError

# ----------------------------------------------------------
# Funciones auxiliares: combinatoria básica
# ----------------------------------------------------------

def factorial(n):
    return 1 if n in (0, 1) else n * factorial(n - 1)

def combination(n, m):
    # Cálculo de combinaciones: "n sobre m"
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))

# ----------------------------------------------------------
# Generación de puntos de referencia 
# ----------------------------------------------------------

def reference_points(npop, nvar):
    # Esta función genera puntos de referencia uniformemente distribuidos
    # en un espacio de 'nvar' dimensiones
    # como direcciones de referencia (Das & Dennis, 1998)

    h1 = 0
    # Se busca el mayor valor de h1 tal que la cantidad de combinaciones posibles
    # (i.e. número de puntos de referencia) no exceda la población deseada
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1

    # Se generan las combinaciones posibles con h1 divisiones
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    
    # Se transforman las combinaciones en coordenadas normalizadas
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) -
              np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1

    # Si aún hay capacidad en la población, se generan puntos adicionales (intermedios)
    if h1 < nvar:
        h2 = 0
        # Se calcula h2 para añadir más puntos sin exceder la población
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1

        # Si h2 > 0, se generan los puntos adicionales
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) -
                           np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            # Se ajustan los puntos para colocarlos en posiciones intermedias
            temp_points = temp_points / 2 + 1 / (2 * nvar)

            # Se añaden a la lista de puntos de referencia original
            points = np.concatenate((points, temp_points), axis=0)

    return points


# ----------------------------------------------------------
# Normalización de objetivos respecto a frontera
# ----------------------------------------------------------

def normalize_objectives(F, front0_indices):
    # Esta función normaliza los vectores objetivo F
    # usando la frontera de Pareto del primer frente (front0_indices)

    # Se obtiene el mínimo global por objetivo
    z_min = F.min(axis=0)

    # Se obtiene el máximo dentro del primer frente para cada objetivo
    z_max = F[front0_indices].max(axis=0)

    # Se normaliza cada vector de objetivos para que sus componentes
    # estén entre 0 y 1 (aproximadamente)
    return (F - z_min) / (z_max - z_min + 1e-12)  # Se agrega 1e-12 para evitar divisiones por cero


# ----------------------------------------------------------
# Fast Non-Dominated Sorting (NSGA-II style)
# ----------------------------------------------------------

def nd_sort(objs):
    # Esta función implementa el ordenamiento rápido no-dominado (Fast Non-Dominated Sorting)
    # Clasifica una población en frentes de Pareto.

    (npop, nobj) = objs.shape  # npop: número de individuos, nobj: número de objetivos

    n = np.zeros(npop, dtype=int)  # Número de individuos que dominan al individuo i
    s = []  # Lista de individuos dominados por cada individuo i
    rank = np.zeros(npop, dtype=int)  # Rango (frente) asignado a cada individuo
    ind = 0  # Índice del frente actual
    pfs = {ind: []}  # Diccionario para almacenar los frentes de Pareto: pfs[0] = [individuos del primer frente]

    # Paso 1: Comparar cada par de individuos i y j
    for i in range(npop):
        s.append([])  # Inicializa lista de dominados por i
        for j in range(npop):
            if i != j:
                # Variables auxiliares para contar la relación entre objetivos de i y j
                less = equal = more = 0
                for k in range(nobj):  # Compara objetivo por objetivo
                    if objs[i, k] < objs[j, k]: less += 1
                    elif objs[i, k] == objs[j, k]: equal += 1
                    else: more += 1

                # Si j domina a i
                if less == 0 and equal != nobj:
                    n[i] += 1
                # Si i domina a j
                elif more == 0 and equal != nobj:
                    s[i].append(j)  # j es dominado por i

        # Si nadie domina a i, entonces pertenece al primer frente
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind  # Se le asigna el rango 0

    # Paso 2: Construir frentes siguientes
    while pfs[ind]:
        pfs[ind + 1] = []  # Inicializa el siguiente frente
        for i in pfs[ind]:  # Para cada individuo en el frente actual
            for j in s[i]:  # Para cada individuo que i domina
                n[j] -= 1  # Disminuye el contador de dominaciones
                if n[j] == 0:  # Si ya nadie domina a j
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1  # Se le asigna el siguiente rango
        ind += 1

    pfs.pop(ind)  # El último frente está vacío, se elimina
    return pfs, rank  # Devuelve el diccionario de frentes y los rangos por individuo


# ----------------------------------------------------------
# Selección basada en nichos para el último frente parcial
# ----------------------------------------------------------

def niching_selection(F_norm, ref_dirs, front_indices, N_remaining):
    """
    Esta función selecciona N_remaining soluciones del último frente parcial usando
    la estrategia de nichos (niching) basada en direcciones de referencia.
    
    Parámetros:
        F_norm : np.ndarray
            Objetivos normalizados (toda la población).
        ref_dirs : np.ndarray
            Direcciones de referencia uniformemente distribuidas.
        front_indices : list[int]
            Índices de las soluciones que pertenecen al último frente.
        N_remaining : int
            Número de soluciones que aún deben ser seleccionadas.

    Retorna:
        selected_indices : list[int]
            Índices de las soluciones seleccionadas para completar la población.
    """

    # Se extraen los vectores de objetivos normalizados correspondientes al último frente
    F_partial = F_norm[front_indices]

    # Se calcula la similitud de coseno entre cada solución y cada dirección de referencia
    cosine = 1 - cdist(F_partial, ref_dirs, metric='cosine')

    # Se calcula la distancia perpendicular desde cada solución a cada dirección
    # La distancia se basa en la proyección del vector normalizado
    norm = np.linalg.norm(F_partial, axis=1).reshape(-1, 1)
    distance = norm * np.sqrt(1 - cosine ** 2)

    # Se asigna cada solución a su dirección de referencia (nicho) más cercana
    assigned_refs = np.argmin(distance, axis=1)
    assigned_distances = np.min(distance, axis=1)

    # Se cuentan cuántas soluciones hay actualmente en cada nicho
    niche_counts = np.zeros(ref_dirs.shape[0], dtype=int)
    for ref in assigned_refs:
        niche_counts[ref] += 1

    # Flags para saber qué soluciones y nichos ya han sido usados
    selected_flags = np.full(len(F_partial), False)  # Soluciones aún no seleccionadas
    ref_flags = np.full(len(ref_dirs), True)         # Direcciones aún disponibles
    selected_indices = []  # Lista final de índices seleccionados

    # Mientras no se haya completado la población
    while len(selected_indices) < N_remaining:
        # Se consideran solo nichos aún disponibles
        candidate_refs = np.where(ref_flags)[0]

        # Se identifica el número mínimo de soluciones en los nichos candidatos
        min_count = np.min(niche_counts[candidate_refs])

        # Se filtran los nichos con ese mínimo número de soluciones
        candidates = candidate_refs[niche_counts[candidate_refs] == min_count]

        # Se escoge aleatoriamente uno de esos nichos candidatos
        chosen_ref = np.random.choice(candidates)

        # Se buscan soluciones aún no seleccionadas que estén asignadas a ese nicho
        sol_idxs = np.where((assigned_refs == chosen_ref) & (~selected_flags))[0]

        if sol_idxs.size > 0:
            # Se elige la más cercana a la dirección de referencia (menor distancia)
            best = sol_idxs[np.argmin(assigned_distances[sol_idxs])]

            # Se marca como seleccionada y se añade a la lista final
            selected_flags[best] = True
            selected_indices.append(front_indices[best])
            niche_counts[chosen_ref] += 1
        else:
            # Si no hay soluciones disponibles en ese nicho, se marca como no usable
            ref_flags[chosen_ref] = False

    return selected_indices


# ----------------------------------------------------------
# Operadores evolutivos: selección, cruce y mutación
# ----------------------------------------------------------

def selection(pop, pc, rank, k=2):
    """
    Realiza la selección por torneo binario sobre la población actual.
    
    Parámetros:
        pop : np.ndarray
            Población actual (npop x nvar).
        pc : float
            Porcentaje de individuos seleccionados para cruzamiento.
        rank : np.ndarray
            Vector de rangos (frentes) para cada individuo.
        k : int
            Tamaño del torneo (por defecto 2: torneo binario).

    Retorna:
        mating_pool : np.ndarray
            Individuos seleccionados para el cruce.
    """
    (npop, nvar) = pop.shape

    # Se calcula el número de individuos seleccionados (debe ser par)
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1  # Asegura que sea par

    # Inicializa el mating pool
    mating_pool = np.zeros((nm, nvar))

    for i in range(nm):
        # Se eligen dos individuos al azar sin reemplazo
        [ind1, ind2] = np.random.choice(npop, k, replace=False)

        # Se selecciona el de mejor frente (menor rango)
        mating_pool[i] = pop[ind1] if rank[ind1] <= rank[ind2] else pop[ind2]

    return mating_pool


def crossover(mating_pool, lb, ub, pc, eta_c):
    """
    Realiza el cruce Simulado Binario (SBX) sobre el mating pool.

    Parámetros:
        mating_pool : np.ndarray
            Individuos seleccionados para reproducción (2n x nvar).
        lb : np.ndarray o float
            Límite inferior de cada variable.
        ub : np.ndarray o float
            Límite superior de cada variable.
        pc : float
            Probabilidad de cruce.
        eta_c : float
            Parámetro de distribución SBX (más alto = hijos más parecidos).

    Retorna:
        offspring : np.ndarray
            Descendencia generada tras el cruce (mismo tamaño que mating_pool).
    """

    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)  # Número de parejas

    # División del mating pool en padres
    parent1, parent2 = mating_pool[:nm], mating_pool[nm:]

    # Inicializa beta para el SBX
    beta = np.zeros((nm, nvar))

    # Genera números aleatorios para determinar beta
    mu = np.random.random((nm, nvar))

    # Determina beta usando el parámetro eta_c
    flag1, flag2 = mu <= 0.5, mu > 0.5
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))

    # Aleatoriza el signo de beta
    beta *= (-1) ** np.random.randint(0, 2, (nm, nvar))

    # Aleatoriamente fuerza beta = 1 para mantener diversidad
    beta[np.random.random((nm, nvar)) < 0.5] = 1

    # Aplica probabilidad de cruce: si no se realiza, beta = 1 → hijos = padres
    beta[np.random.random((nm, nvar)) > pc] = 1

    # Se generan los hijos usando la fórmula del SBX
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2

    # Se concatenan los hijos y se asegura que respeten los límites
    offspring = np.clip(np.concatenate((offspring1, offspring2), axis=0), lb, ub)

    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    """
    Aplica mutación polinómica a una población real-valuada.

    Parámetros:
        pop : np.ndarray
            Población actual (npop x nvar).
        lb : float o np.ndarray
            Límite inferior para cada variable.
        ub : float o np.ndarray
            Límite superior para cada variable.
        pm : float
            Probabilidad de mutación (típicamente 1.0).
        eta_m : float
            Parámetro de distribución de la mutación (más alto = perturbaciones pequeñas).

    Retorna:
        pop : np.ndarray
            Población mutada (respeta límites).
    """
    (npop, nvar) = pop.shape

    # Asegura que lb y ub tengan la misma forma que la población
    lb, ub = np.tile(lb, (npop, 1)), np.tile(ub, (npop, 1))

    # Determina en qué sitios se aplicará la mutación (cada variable con probabilidad pm/nvar)
    site = np.random.random((npop, nvar)) < pm / nvar

    # Números aleatorios para decidir cómo aplicar la mutación
    mu = np.random.random((npop, nvar))

    # Calcula la distancia de cada gen al límite inferior y superior (normalizada)
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)

    # Casos donde mu <= 0.5 → se usa una fórmula distinta
    temp = site & (mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (
        (2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
    )

    # Casos donde mu > 0.5 → otra fórmula
    temp = site & (mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (
        1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1))
    )

    # Se asegura que los valores mutados no sobrepasen los límites definidos
    return np.clip(pop, lb, ub)


# ----------------------------------------------------------
# Selección ambiental NSGA-III
# ----------------------------------------------------------

def environmental_selection(pop, objs, zmin, npop, V):
    """
    Selecciona la siguiente generación de tamaño `npop` usando la lógica de NSGA-III.

    Parámetros:
        pop : np.ndarray
            Población actual (individuos).
        objs : np.ndarray
            Valores de las funciones objetivo.
        zmin : np.ndarray
            Punto ideal (mínimos por objetivo).
        npop : int
            Tamaño deseado de la nueva población.
        V : np.ndarray
            Direcciones de referencia (nichos).

    Retorna:
        pop_sel : np.ndarray
            Individuos seleccionados.
        objs_sel : np.ndarray
            Objetivos correspondientes a los seleccionados.
        rank_sel : np.ndarray
            Rango (frente) de los individuos seleccionados.
    """

    # Ordenamiento no dominado
    pfs, rank = nd_sort(objs)
    nobj = objs.shape[1]

    # Marca los individuos que caben completamente en los frentes dominados
    selected = np.full(pop.shape[0], False)
    ind = 0
    while np.sum(selected) + len(pfs[ind]) <= npop:
        selected[pfs[ind]] = True
        ind += 1

    # Número de individuos faltantes
    K = npop - np.sum(selected)

    # Se toman los objetivos seleccionados y los del último frente parcial
    objs1, objs2 = objs[selected], objs[pfs[ind]]

    # Se concatenan y se normalizan restando el punto ideal
    t_objs = np.concatenate((objs1, objs2), axis=0) - zmin

    # -------------------------------
    # Estimación de la frontera de referencia (normalización)
    # -------------------------------
    # Se buscan puntos extremos usando pesos casi unitarios
    extreme = np.zeros(nobj)
    w = 1e-6 + np.eye(nobj)  # Pesos casi unitarios para cada objetivo
    for i in range(nobj):
        extreme[i] = np.argmin(np.max(t_objs / w[i], axis=1))  # ASF: Achievement Scalarizing Function

    try:
        # Se intenta construir el hiperplano que pasa por los extremos
        hyperplane = np.linalg.inv(t_objs[extreme.astype(int)]) @ np.ones((nobj, 1))
        a = 1 / hyperplane if not np.any(hyperplane == 0) else np.max(t_objs, axis=0)
    except LinAlgError:
        # Si no es posible invertir (singular), se usa como máximo por objetivo
        a = np.max(t_objs, axis=0)

    # Se normalizan los objetivos dividiendo por el valor del hiperplano
    t_objs /= a.reshape(1, nobj)

    # -------------------------------
    # Asociación a nichos (direcciones de referencia)
    # -------------------------------
    cosine = 1 - cdist(t_objs, V, 'cosine')  # Similitud de coseno
    distance = np.linalg.norm(t_objs, axis=1).reshape(-1, 1) * np.sqrt(1 - cosine ** 2)

    # Se obtiene la distancia mínima y la asociación (índice del nicho más cercano)
    dis = np.min(distance, axis=1)
    association = np.argmin(distance, axis=1)

    # Contar cuántos individuos ya hay en cada nicho (solo los seleccionados)
    temp_rho = dict(Counter(association[:objs1.shape[0]]))
    rho = np.zeros(V.shape[0])
    for k in temp_rho.keys():
        rho[k] = temp_rho[k]

    # Flags para soluciones y nichos del último frente parcial
    choose = np.full(objs2.shape[0], False)
    v_choose = np.full(V.shape[0], True)

    # -------------------------------
    # Selección nichada para completar la población
    # -------------------------------
    while np.sum(choose) < K:
        temp = np.where(v_choose)[0]  # Direcciones disponibles
        jmin = np.where(rho[temp] == np.min(rho[temp]))[0]  # Las menos pobladas
        j = temp[np.random.choice(jmin)]  # Elige una dirección aleatoria entre las menos pobladas

        # Soluciones en el último frente asociadas a esa dirección
        I = np.where((~choose) & (association[objs1.shape[0]:] == j))[0]

        if I.size > 0:
            # Si el nicho está vacío, elige la más cercana, si no, elige aleatoriamente
            s = np.argmin(dis[objs1.shape[0] + I]) if rho[j] == 0 else np.random.randint(I.size)
            choose[I[s]] = True
            rho[j] += 1
        else:
            v_choose[j] = False  # Si ya no hay candidatos en ese nicho, se descarta

    # Se actualiza la máscara de seleccionados con los elegidos del último frente
    selected[np.array(pfs[ind])[choose]] = True

    # Se devuelve la población y objetivos seleccionados, junto con sus rangos
    return pop[selected], objs[selected], rank[selected]

def cal_obj(pop_lambdas, nobj):
    results = []
    for lambdas in pop_lambdas:
        lambda1, lambda2, lambda3 = lambdas

        # Entorno modificado con estos lambdas
        env_kwargs = {
            "hmax": 1000,
            "initial_amount": 1_000_000,
            "buy_cost_pct": [0.001] * stock_dimension,
            "sell_cost_pct": [0.001] * stock_dimension,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "esg_dict": ESG_dict
        }

        # === Entrenamiento PPO ===
        env_train = StockTradingMultiEnv(df=train, **env_kwargs)
        agent = DRLAgent(env=env_train)
        model = agent.get_model("ppo")
        trained_model = agent.train_model(model, total_timesteps=1000)

        # === Evaluación ===
        env_eval = DummyVecEnv([lambda: StockTradingMultiEnv(df=trade, **env_kwargs)])
        obs = env_eval.reset()
        r1_list, r2_list, r3_list = [], [], []

        for _ in range(len(trade.date.unique()) - 1):
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, done, info = env_eval.step(action)
            r1_list.append(info[0]["r1"])
            r2_list.append(info[0]["r2"])
            r3_list.append(info[0]["r3"])
            if done:
                break

        results.append([np.mean(r1_list), np.mean(r2_list), np.mean(r3_list)])

    return np.array(results)


# ----------------------------------------------------------
# Función principal
# ----------------------------------------------------------

def main(npop, iter, lb, ub, nobj=3, pc=1, pm=1, eta_c=30, eta_m=20):
    """
    Ejecuta el algoritmo NSGA-III desde cero sobre el problema WFG1.

    Parámetros:
        npop : int
            Tamaño de la población.
        iter : int
            Número de generaciones a ejecutar.
        lb, ub : np.ndarray
            Límite inferior y superior de las variables de decisión.
        nobj : int
            Número de objetivos del problema.
        pc : float
            Probabilidad de cruce.
        pm : float
            Probabilidad de mutación.
        eta_c : float
            Parámetro de distribución del cruce SBX.
        eta_m : float
            Parámetro de distribución de la mutación polinómica.
    """

    nvar = len(lb)  # Número de variables de decisión

    # Población inicial generada aleatoriamente
    pop = np.random.uniform(lb, ub, (npop, nvar))

    # Evaluación inicial de objetivos
    objs = cal_obj(pop, nobj)

    # Generación de direcciones de referencia (nichos)
    V = reference_points(npop, nobj)

    # Inicialización del punto ideal
    zmin = np.min(objs, axis=0)

    # Primer ordenamiento no dominado
    pfs, rank = nd_sort(objs)

    # Bucle principal evolutivo
    for t in range(iter):
        if (t + 1) % 50 == 0:
            print(f'Iteration: {t + 1} completed.')

        # Selección por torneo binario
        mating_pool = selection(pop, pc, rank)

        # Cruce SBX
        off = crossover(mating_pool, lb, ub, pc, eta_c)

        # Mutación polinómica
        off = mutation(off, lb, ub, pm, eta_m)

        # Evaluación de la descendencia
        off_objs = cal_obj(off, nobj)

        # Actualización del punto ideal
        zmin = np.minimum(zmin, np.min(off_objs, axis=0))

        # Selección ambiental (NSGA-III)
        pop, objs, rank = environmental_selection(
            np.vstack((pop, off)),
            np.vstack((objs, off_objs)),
            zmin,
            npop,
            V
        )

    # ---------------------------------------------
    # Resultados finales: extraer el frente de Pareto
    # ---------------------------------------------
    pf = objs[rank == 0]  # Solo individuos del frente 0

    # Calcular resumen estadístico de los objetivos del frente de Pareto
    resumen_estadistico = {
        "Objetivo": [f"Objetivo {i+1}" for i in range(pf.shape[1])],
        "Promedio": np.mean(pf, axis=0),
        "Desviación Estándar": np.std(pf, axis=0)
    }

    # Imprimir resumen estadístico
    print("\nResumen Estadístico del Frente de Pareto:")
    for i in range(pf.shape[1]):
        print(f"{resumen_estadistico['Objetivo'][i]} → Promedio: {resumen_estadistico['Promedio'][i]:.4f}, "
              f"Desviación Est.: {resumen_estadistico['Desviación Estándar'][i]:.4f}")

    # ---------------------------------------------
    # Visualización del frente (3D)
    # ---------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 45)  # Ángulo de vista
    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], color='magenta')
    ax.set_xlabel('Objetivo 1')
    ax.set_ylabel('Objetivo 2')
    ax.set_zlabel('Objetivo 3')
    plt.title('Frente de Pareto para WFG1')
    plt.savefig('NSGA3_WFG1.png')  # Guarda la imagen
    plt.show()

# ----------------------------------------------------------
# Evaluación del problema WFG1
# ----------------------------------------------------------


lb = np.array([0.1, 0.1, 0.1])  # límites inferiores de lambda1, lambda2, lambda3
ub = np.array([2.0, 1.0, 3.0])  # límites superiores (ajusta si lo deseas)
main(npop=10, iter=3, lb=lb, ub=ub, nobj=3)

