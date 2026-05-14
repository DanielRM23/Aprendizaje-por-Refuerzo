import subprocess
import os

# ----------------------------
# ETAPA 1: Métricas
# ----------------------------
print("\n Paso 1: Ejecutando extracción y evaluación de métricas...\n")

archivo1 = "get_metrics.py" #cambiar el nombre del archivo 
archivo2 = "Replicado.py" #cambiar el nombre del archivo

try:
    
    subprocess.run(["python", archivo1], check=True)
except subprocess.CalledProcessError:
    print(f"Error al ejecutar {archivo1}")
    exit(1)

# ----------------------------
# ETAPA 2: Graficar los resultados
# ----------------------------
print("\n Paso 2: Generando gráficas comparativas...\n")

try:
    subprocess.run(["python", archivo2], check=True)
except subprocess.CalledProcessError:
    print(f"Error al ejecutar {archivo2}")
    exit(1)

print("\n Proceso completado correctamente. Revisa las gráficas en la carpeta 'graficas_metricas'.")
