from MDP_FrozenLake import FrozenLake
from Iteraciones import iteracion_valor, iteracion_politica

# Crear el entorno
ambiente = FrozenLake()

# Ejecutar Iteración de Valor
print("Iteración de Valor:")
V_optimo = iteracion_valor(ambiente)
ambiente.mostrar_valor_V(V_optimo)

# Ejecutar Iteración de Política
print("\nIteración de Política:")
politica_optima, V_optimo = iteracion_politica(ambiente)
ambiente.mostrar_politica_optima(politica_optima)
ambiente.mostrar_valor_V(V_optimo)