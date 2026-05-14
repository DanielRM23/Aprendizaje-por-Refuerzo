import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
ARCHIVO_CSV = "output_metrics.csv"
CARPETA_SALIDA = "graficas_metricas"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ----------------------------
# CARGA DE DATOS
# ----------------------------
try:
    df = pd.read_csv(ARCHIVO_CSV)
except FileNotFoundError:
    print(f"No se encontró el archivo: {ARCHIVO_CSV}")
    exit()


# ----------------------------
# GRAFICADO POR MÉTRICA
# ----------------------------
juegos = df["game"].unique()
metodos = df["method"].unique()
metricas = ["precision", "recall", "f1_score", "iou"]

for metrica in metricas:
    plt.figure(figsize=(8, 5))

    # Extrae valores por método
    for metodo in metodos:
        subset = df[df["method"] == metodo]
        valores = subset[metrica].values
        plt.bar([j + (0.2 if metodo == "VEM" else -0.2) for j in range(len(juegos))],
                valores, width=0.4, label=metodo)

    plt.xticks(range(len(juegos)), juegos)
    plt.title(f"Comparación de {metrica.upper()} entre REM y VEM")
    plt.ylabel(metrica)
    plt.xlabel("Juego")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()

    # Guardar imagen
    ruta_imagen = os.path.join(CARPETA_SALIDA, f"{metrica}_comparacion.png")
    plt.savefig(ruta_imagen)
    print(f"Guardado: {ruta_imagen}")
    plt.close()

print("\nGráficas generadas en la carpeta:", CARPETA_SALIDA)
