# Tarea 3, Rojo Mata Daniel

---

## Algoritmos Implementados

- **DDQN** (Double Deep Q-Network)
- **NES** (Natural Evolution Strategies)
- **PPO** (Proximal Policy Optimization)

Cada algoritmo fue implementado y entrenado tanto en el entorno **MinAtar BreakOut** como en **ALE/Pong-v5** utilizando `Gymnasium`.

---

## Visualizaciones

Cada script genera una gráfica `.png` correspondiente, donde se puede observar la evolución de la recompensa promedio por episodio durante el entrenamiento.

---

## Requisitos

Este proyecto fue desarrollado con **Python 3.8+** y requiere las siguientes librerías:

- Gymnasium (`gymnasium[atari]`)
- MinAtar
- OpenCV (`cv2`)
- PyTorch
- Matplotlib
- NumPy

### Instalación

Puedes instalar todos los paquetes necesarios ejecutando los siguientes comandos:

pip install gymnasium[atari]
pip install minatar
pip install opencv-python
pip install torch
pip install matplotlib
pip install numpy 


## Ejecución

Para entrenar un agente, solo ejecuta el archivo correspondiente. Por ejemplo:

python DDQN_BreakOut.py

