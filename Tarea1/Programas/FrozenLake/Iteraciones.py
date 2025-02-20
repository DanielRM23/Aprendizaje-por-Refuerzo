import numpy as np

# Algoritmo de Iteración de Valor:
# Esta función calcula la función de valor óptima V(s) para cada estado del ambiente.
def iteracion_valor(ambiente, factor_descuento=0.99, theta=1e-6):
    # Inicializa V(s) para cada estado a cero.
    V = np.zeros((ambiente.tamano, ambiente.tamano))
    
    # El bucle while se ejecuta hasta que la actualización de V sea menor que theta (criterio de convergencia).
    while True:
        delta = 0  # Variable para llevar la máxima diferencia en V en cada iteración.
        
        # Se recorren todos los estados del ambiente (usando índices i y j).
        for i in range(ambiente.tamano):
            for j in range(ambiente.tamano):
                # Si el estado es un "hoyo" (valor 1 en la matriz), se omite el cálculo ya que no se puede actuar.
                if ambiente.matriz[i][j] == 1:
                    continue
                    
                v = V[i][j]  # Se guarda el valor actual para comparar luego la diferencia.
                
                # Calcula el valor resultante para cada acción (se asume que hay 4 acciones posibles).
                valores_accion = []
                for accion in range(4):
                    # Se obtiene el nuevo estado que se alcanzaría aplicando la acción.
                    nuevo_estado = ambiente.actualizar_posicion(i, j, accion)
                    # Se obtiene la recompensa y otros resultados asociados al nuevo estado.
                    recompensa, _ = ambiente.determinar_resultado(nuevo_estado)
                    # Se calcula el valor de la acción usando la ecuación de Bellman:
                    # Q(s, a) = recompensa(s,a) + factor_descuento * V(s')
                    valores_accion.append(recompensa + factor_descuento * V[nuevo_estado[0]][nuevo_estado[1]])
                    
                # Se actualiza V(s) en el estado actual tomando el valor máximo entre las acciones.
                V[i][j] = max(valores_accion)
                # Se actualiza delta con la diferencia máxima encontrada.
                delta = max(delta, abs(v - V[i][j]))
                
        # Si la mayor diferencia es menor al umbral theta, se considera que V ha convergido.
        if delta < theta:
            break
            
    return V

# Algoritmo de Evaluación de Política:
# Dada una política fija (que asigna una acción a cada estado), se calcula la función de valor V(s)
# correspondiente a esa política.
def evaluacion_politica(ambiente, politica, factor_descuento=0.99, theta=1e-6):
    V = np.zeros((ambiente.tamano, ambiente.tamano))
    
    # Se itera hasta la convergencia de V(s) con la política actual.
    while True:
        delta = 0
        for i in range(ambiente.tamano):
            for j in range(ambiente.tamano):
                if ambiente.matriz[i][j] == 1:  # Si es un hoyo, se omite.
                    continue
                    
                v = V[i][j]  # Valor actual del estado.
                # Se obtiene la acción asignada por la política en el estado (i,j).
                accion = politica[i][j]
                # Se calcula el nuevo estado tras aplicar la acción.
                nuevo_estado = ambiente.actualizar_posicion(i, j, accion)
                # Se obtiene la recompensa asociada a la transición.
                recompensa, _ = ambiente.determinar_resultado(nuevo_estado)
                # Se actualiza el valor del estado usando la ecuación de Bellman para la política fija:
                V[i][j] = recompensa + factor_descuento * V[nuevo_estado[0]][nuevo_estado[1]]
                delta = max(delta, abs(v - V[i][j]))
        if delta < theta:
            break
    return V

# Algoritmo de Mejora de Política:
# Con la función de valor V(s) calculada, se determina la mejor acción para cada estado,
# actualizando la política.
def mejora_politica(ambiente, V, factor_descuento=0.99):
    # Inicializa la política (en cada estado se asigna inicialmente la acción 0).
    politica = np.zeros((ambiente.tamano, ambiente.tamano), dtype=int)
    
    # Se recorre cada estado.
    for i in range(ambiente.tamano):
        for j in range(ambiente.tamano):
            if ambiente.matriz[i][j] == 1:  # Si es un hoyo, se salta.
                continue
                
            # Se calculan los valores resultantes de aplicar cada acción posible.
            valores_accion = []
            for accion in range(4):
                nuevo_estado = ambiente.actualizar_posicion(i, j, accion)
                recompensa, _ = ambiente.determinar_resultado(nuevo_estado)
                valores_accion.append(recompensa + factor_descuento * V[nuevo_estado[0]][nuevo_estado[1]])
                
            # Se selecciona la acción que maximiza el valor.
            politica[i][j] = np.argmax(valores_accion)
    return politica

# Algoritmo de Iteración de Política:
# Combina la evaluación y mejora de política iterativamente para encontrar la política óptima.
def iteracion_politica(ambiente, factor_descuento=0.99, theta=1e-6):
    # Se inicializa la política (por ejemplo, con acción 0 en cada estado).
    politica = np.zeros((ambiente.tamano, ambiente.tamano), dtype=int)
    
    while True:
        # Se evalúa la política actual para obtener la función de valor V(s).
        V = evaluacion_politica(ambiente, politica, factor_descuento, theta)
        # Se mejora la política usando V(s).
        nueva_politica = mejora_politica(ambiente, V, factor_descuento)
        
        # Si la política no cambia tras la mejora, se ha alcanzado la convergencia.
        if np.array_equal(politica, nueva_politica):
            break
            
        # Se actualiza la política para la siguiente iteración.
        politica = nueva_politica
        
    # Se devuelve la política óptima y su correspondiente función de valor.
    return politica, V
