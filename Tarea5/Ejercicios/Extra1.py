# Importamos las librerías necesarias
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from collections import deque
import random

# Creamos una clase que adapta CartPole a un entorno multiobjetivo
# Lo que se hace es agregar una nueva recompensa: la inclinación del palo 
class CartPoleMultiobjetivo(gym.Wrapper):
    def __init__(self, entorno):
        # Inicializamos el wrapper con el entorno base
        super().__init__(entorno)

    def step(self, accion):
        # Ejecutamos la acción en el entorno base y obtenemos la transición
        observacion, recompensa, terminado, truncado, info = self.env.step(accion)

        # Primer objetivo: recompensa original (tiempo de supervivencia)
        recompensa_1 = recompensa

        # Segundo objetivo: castigo por la inclinación del palo (ángulo distinto de 0)
        angulo = observacion[2]
        recompensa_2 = -abs(angulo)

        # Combinamos ambas recompensas en un vector
        recompensa_vector = np.array([recompensa_1, recompensa_2])

        # Retornamos la observación, el vector de recompensas y los indicadores de finalización
        return observacion, recompensa_vector, terminado, truncado, info

    def reset(self, **kwargs):
        # Reiniciamos el entorno
        return self.env.reset(**kwargs)


# Función para simular un episodio con el entorno dado
def simular_episodio(entorno, es_multiobjetivo=True):
    observacion, _ = entorno.reset()
    terminado = False
    historial_r1, historial_r2 = [], []

    while not terminado:
        # Elegimos una acción aleatoria
        accion = entorno.action_space.sample()
        observacion, recompensas, fin, truncado, _ = entorno.step(accion)

        if es_multiobjetivo:
            historial_r1.append(recompensas[0])
            historial_r2.append(recompensas[1])
        else:
            historial_r1.append(recompensas)
            historial_r2.append(0)  # No hay segunda recompensa

        terminado = fin or truncado

    # Devolvemos las recompensas acumuladas
    return np.cumsum(historial_r1), np.cumsum(historial_r2)



class RedPDQN(nn.Module):
    def __init__(self, dim_obs, dim_act, dim_pref):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs + dim_pref, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim_act * dim_pref)
        )
        self.dim_act = dim_act
        self.dim_pref = dim_pref

    def forward(self, obs, omega):
        x = torch.cat([obs, omega], dim=-1)
        q = self.net(x)
        return q.view(-1, self.dim_act, self.dim_pref)  # Q(s,a) ∈ ℝ^{acciones × objetivos}


def seleccionar_accion(modelo, obs, omega):
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    omega = torch.tensor(omega, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_vect = modelo(obs, omega)  # (1, acciones, objetivos)
        q_scalar = torch.sum(q_vect * omega.unsqueeze(1), dim=2)  # (1, acciones)
        return torch.argmax(q_scalar).item()



class ReplayBuffer:
    def __init__(self, capacidad=10000):
        self.buffer = deque(maxlen=capacidad)

    def guardar(self, transicion):
        self.buffer.append(transicion)

    def samplear(self, batch_size):
        muestras = random.sample(self.buffer, batch_size)
        estados, omegas, acciones, recompensas, estados_sig, dones = zip(*muestras)
        return (np.array(estados), np.array(omegas), np.array(acciones),
                np.array(recompensas), np.array(estados_sig), np.array(dones))

    def __len__(self):
        return len(self.buffer)


def entrenar_pdqn(env, modelo, buffer, optimizador, episodios=500, gamma=0.99, batch_size=64):
    recompensas_totales = []

    for ep in range(episodios):
        estado, _ = env.reset()
        omega = np.random.dirichlet([1.0, 1.0])  # Información experta simulada

        total_r1, total_r2 = 0.0, 0.0
        terminado = False

        while not terminado:
            accion = seleccionar_accion(modelo, estado, omega)
            estado_sig, recomp_vec, fin, truncado, _ = env.step(accion)

            # Guardamos la transición original
            buffer.guardar((estado, omega, accion, recomp_vec, estado_sig, fin or truncado))

            # HER: reusamos la transición con ω aleatorios
            for _ in range(10):  # puedes ajustar 4 → número de muestras HER por transición
                omega_her = np.random.dirichlet([1.0, 1.0])
                buffer.guardar((estado, omega_her, accion, recomp_vec, estado_sig, fin or truncado))


            estado = estado_sig
            total_r1 += recomp_vec[0]
            total_r2 += recomp_vec[1]
            terminado = fin or truncado

            # Entrenar si hay suficientes muestras
            if len(buffer) >= batch_size:
                estados, omegas, acciones, recompensas, estados_sig, dones = buffer.samplear(batch_size)

                estados = torch.tensor(estados, dtype=torch.float32)
                omegas = torch.tensor(omegas, dtype=torch.float32)
                acciones = torch.tensor(acciones, dtype=torch.long).unsqueeze(1)
                recompensas = torch.tensor(recompensas, dtype=torch.float32)
                estados_sig = torch.tensor(estados_sig, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                # Obtener Q actuales
                q_vals = modelo(estados, omegas)
                q_sa = torch.sum(q_vals.gather(1, acciones.unsqueeze(-1).expand(-1, 1, modelo.dim_pref)).squeeze(1) * omegas, dim=1)

                # Calcular Q_target
                with torch.no_grad():
                    #q_sig = modelo(estados_sig, omegas)
                    q_sig = modelo_target(estados_sig, omegas)
                    q_sig_scalar = torch.sum(q_sig * omegas.unsqueeze(1), dim=2)
                    max_q_sig = torch.max(q_sig_scalar, dim=1)[0]
                    y = torch.sum(recompensas * omegas, dim=1) + gamma * (1 - dones.squeeze()) * max_q_sig

                # MSE Loss
                loss = nn.MSELoss()(q_sa, y)

                optimizador.zero_grad()
                loss.backward()
                optimizador.step()

        recompensas_totales.append([total_r1, total_r2])
        # Soft update de la red objetivo
        tau = 0.005  # Puedes ajustar este valor si quieres experimentar
        for param, target_param in zip(modelo.parameters(), modelo_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if ep % 100 == 0:
            print(f"Episodio {ep}: R1={total_r1:.1f}, R2={total_r2:.3f}")

    return np.array(recompensas_totales)



# Crear entorno
env = CartPoleMultiobjetivo(gym.make("CartPole-v1"))

# Dimensiones
dim_obs = env.observation_space.shape[0]       # 4 para CartPole
dim_act = env.action_space.n                   # 2 acciones
dim_pref = 2                                   # 2 objetivos

# Crear red y optimizador
modelo = RedPDQN(dim_obs, dim_act, dim_pref)
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

modelo_target = RedPDQN(dim_obs, dim_act, dim_pref)
modelo_target.load_state_dict(modelo.state_dict())
modelo_target.eval()  # No se entrena directamente


# Buffer
buffer = ReplayBuffer(capacidad=50000)

# Entrenamos el agente
recompensas_entrenamiento = entrenar_pdqn(env, modelo, buffer, optimizador, episodios=1000)


def simular_episodio(entorno, modelo, omega):
    observacion, _ = entorno.reset()
    terminado = False
    historial_r1, historial_r2 = [], []

    while not terminado:
        accion = seleccionar_accion(modelo, observacion, omega)
        observacion, recompensas, fin, truncado, _ = entorno.step(accion)
        historial_r1.append(recompensas[0])
        historial_r2.append(recompensas[1])
        terminado = fin or truncado

    return np.array(historial_r1), np.array(historial_r2)


# Evaluamos con distintas preferencias
plt.figure(figsize=(8, 5))
for omega in [[1.0, 0.0], [0.5, 0.5], [0.1, 0.9]]:
    r1, r2 = simular_episodio(env, modelo, np.array(omega))
    plt.plot(np.cumsum(r1), np.cumsum(r2), label=f"ω = {omega}")

plt.xlabel("Pasos")
plt.ylabel("Recompensas acumuladas")
plt.title("Evaluación del modelo con distintas preferencias")
plt.legend()
plt.grid(True)
plt.savefig("evaluacion_pdqn.png")
plt.show()