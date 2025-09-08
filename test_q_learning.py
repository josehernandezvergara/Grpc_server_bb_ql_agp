# test_q_learning.py
# Script de prueba para validar Q-learning en el simulador (solo Python, sin Unity).
# Usa las clases definidas en server.py (WarehouseModel, WorkerAgent, qlearn).
#
# Genera tres gráficas: reward_per_step.png, epsilon.png, deliveries.png
# Opciones de CLI: --agents N, --objects M, --steps S, --inference (quita exploración)

import argparse
import time
from collections import deque
import os
import sys

import matplotlib.pyplot as plt

# Importa tu server.py (asegúrate que no arranca el servidor gRPC al importar)
import server

def run_test(agents=3, objects=5, steps=5000, save_plots=True, inference=False, log_every=500):
    """
    Ejecuta el entrenamiento online por 'steps' pasos.
    - agents: número de agentes
    - objects: número de cajas
    - steps: número total de pasos
    - save_plots: guardar PNGs al final
    - inference: si True, fuerza epsilon=0 (explotación)
    - log_every: cada cuantos pasos imprimir resumen
    """

    # Seguridad: si hay un fichero qtable corrupto y quieres empezar limpio,
    # renómbralo fuera de carpeta antes de correr este script.
    print("Inicializando prueba: agents={}, objects={}, steps={}, inference={}".format(
        agents, objects, steps, inference
    ))

    # Crear modelo
    params = {"agents": agents, "objects": objects, "steps": 0}
    model = server.WarehouseModel(params)
    model.setup()
    ag = model.workers[0]
    bx = model.boxes[0]
    bx.grid_pos = (ag.grid_pos[0], ag.grid_pos[1])
    bx.pos = server.grid_to_world(bx.grid_pos)
    target = (min(server.GRID-1, ag.grid_pos[0]+2), ag.grid_pos[1])
    model.blackboard.assign_task_to_agent(ag.id, bx.id, target)
    print("DEBUG: Forzada Box", bx.id, "en", bx.grid_pos, "para Agent", ag.id)

    # Forzamos cargador en el centro (útil durante la prueba)
    try:
        model_center = (server.GRID // 2, server.GRID // 2)
        server.CHARGERS[:] = [model_center]
    except Exception:
        pass

    # Si queremos evaluar (inferencia), forzamos epsilon a 0
    if inference:
        print("Modo INFERENCE: epsilon -> 0")
        server.qlearn.epsilon = 0.0

    # Estadísticas
    rewards_per_step = []
    eps_history = []
    deliveries_history = []
    avg_window = deque(maxlen=200)

    cumulative_reward = 0.0

    # Helpers
    def completed_tasks(bb):
        # alternativa: usar model.total_deliveries
        return model.total_deliveries

    # Main loop
    start_time = time.time()
    for step in range(1, steps + 1):
        step_reward = 0.0

        # Ejecutar paso por agente (usamos ag.step() para recolectar rewards por agente)
        for ag in model.workers:
            info = ag.step()
            if info is not None:
                step_reward += info.get("reward", 0.0)

        # Actualizar contador y guardados periódicos (imitando WarehouseModel.step)
        model.step_counter += 1
        if model.step_counter % server.SAVE_EVERY == 0:
            print(f"[SAVE] Paso {model.step_counter}, guardando Q-table...")
            server.qlearn.save()

        cumulative_reward += step_reward
        rewards_per_step.append(step_reward)
        eps_history.append(server.qlearn.epsilon)
        deliveries_history.append(model.total_deliveries)
        avg_window.append(step_reward)

        # Logs periódicos
        if step % log_every == 0 or step == 1:
            avg_recent = sum(avg_window) / len(avg_window) if len(avg_window) > 0 else 0.0
            print(f"[{step}/{steps}] reward_step={step_reward:.2f} avg_recent={avg_recent:.2f} eps={server.qlearn.epsilon:.4f} deliveries={model.total_deliveries}")

    # Guardar Q final
    print("Entrenamiento finalizado. Guardando Q-table...")
    server.qlearn.save()

    elapsed = time.time() - start_time
    print(f"Tiempo total: {elapsed:.1f}s  Recompensa acumulada: {cumulative_reward:.2f}  Entregas: {model.total_deliveries}")
    print("Tamaño Q-table:", len(server.qlearn.Q))

    # Guardar figuras
    if save_plots:
        try:
            # Reward por paso
            plt.figure(figsize=(10,4))
            plt.plot(rewards_per_step, linewidth=0.6)
            plt.xlabel("Paso")
            plt.ylabel("Reward por paso")
            plt.title("Reward por paso (entrenamiento)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("reward_per_step.png", dpi=150)
            plt.close()

            # Epsilon
            plt.figure(figsize=(8,3))
            plt.plot(eps_history)
            plt.xlabel("Paso")
            plt.ylabel("Epsilon")
            plt.title("Epsilon (exploración)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("epsilon.png", dpi=150)
            plt.close()

            # Deliveries
            plt.figure(figsize=(8,3))
            plt.plot(deliveries_history)
            plt.xlabel("Paso")
            plt.ylabel("Entregas acumuladas")
            plt.title("Entregas durante entrenamiento")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("deliveries.png", dpi=150)
            plt.close()

            print("Gráficas guardadas: reward_per_step.png, epsilon.png, deliveries.png")
        except Exception as e:
            print("Error guardando gráficas:", e)

    return {
        "model": model,
        "rewards": rewards_per_step,
        "eps": eps_history,
        "deliveries": deliveries_history,
        "time_s": elapsed
    }

def main():
    parser = argparse.ArgumentParser(description="Test Q-learning (solo Python) para el simulador de AGVs")
    parser.add_argument("--agents", type=int, default=3, help="Número de agentes")
    parser.add_argument("--objects", type=int, default=5, help="Número de cajas/objetos")
    parser.add_argument("--steps", type=int, default=5000, help="Número de pasos de entrenamiento")
    parser.add_argument("--no-plots", action="store_true", help="No guardar gráficos")
    parser.add_argument("--inference", action="store_true", help="Modo inference (epsilon=0)")
    parser.add_argument("--log-every", type=int, default=500, help="Cada cuántos pasos imprimir log")
    args = parser.parse_args()

    result = run_test(agents=args.agents, objects=args.objects, steps=args.steps,
                      save_plots=(not args.no_plots), inference=args.inference, log_every=args.log_every)

    print("Resultado:", {k:v for k,v in result.items() if k in ("time_s",)})

if __name__ == "__main__":
    main()
