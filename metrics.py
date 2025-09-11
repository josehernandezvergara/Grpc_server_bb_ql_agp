"""
metrics.py
modulo independiente para recolectar y visualizar metricas de entrenamiento

objetivo:
- ser importado desde scripts de entrenamiento (por ejemplo test_q_learning.py)
- no modifica ningun archivo existente del proyecto
- provee un collector simple y funciones para guardar graficas/CSV

uso basico (en tu script de entrenamiento):
from metrics import MetricsCollector
mc = MetricsCollector()
... dentro del bucle de entrenamiento ...
mc.record(step=step, reward=step_reward, epsilon=qlearn.epsilon, deliveries=model.total_deliveries)
al final: mc.save_plots('outdir') ; mc.save_csv('outdir/metrics.csv')

nota: matplotlib es requerida para guardar graficas (esta en requirements.txt del repo)
"""

from typing import List, Optional
import os
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MetricsCollector:
    """coleccion simple de metricas por paso.

    atributos principales:
    - steps: lista de indices de paso
    - rewards: lista de reward total por paso
    - eps: lista de epsilon
    - deliveries: lista de entregas acumuladas
    - meta: diccionario libre para metrics extra
    """

    def __init__(self):
        self.steps: List[int] = []
        self.rewards: List[float] = []
        self.eps: List[float] = []
        self.deliveries: List[int] = []
        self.meta = {}

    def record(self, step: int, reward: float, epsilon: float, deliveries: int):
        """registra las metricas de un paso.

        - step: indice del paso (int)
        - reward: reward agregado en el paso (float)
        - epsilon: valor actual de epsilon (float)
        - deliveries: numero de entregas acumuladas (int)
        """
        self.steps.append(int(step))
        self.rewards.append(float(reward))
        self.eps.append(float(epsilon))
        self.deliveries.append(int(deliveries))

    def save_csv(self, path: str):
        """guarda un csv con las series registradas"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'reward', 'epsilon', 'deliveries'])
            for s, r, e, d in zip(self.steps, self.rewards, self.eps, self.deliveries):
                w.writerow([s, r, e, d])

    def save_plots(self, outdir: str = '.', prefix: Optional[str] = ''):
        """genera y guarda tres graficas: reward por paso, epsilon, deliveries acumuladas"""
        os.makedirs(outdir, exist_ok=True)

        # reward por paso
        try:
            plt.figure(figsize=(10,4))
            plt.plot(self.steps, self.rewards, linewidth=0.6)
            plt.xlabel('paso')
            plt.ylabel('reward por paso')
            plt.title('reward por paso')
            plt.grid(True)
            plt.tight_layout()
            fname = os.path.join(outdir, f"{prefix}reward_per_step.png")
            plt.savefig(fname, dpi=150)
            plt.close()
        except Exception:
            pass

        # epsilon
        try:
            plt.figure(figsize=(8,3))
            plt.plot(self.steps, self.eps)
            plt.xlabel('paso')
            plt.ylabel('epsilon')
            plt.title('epsilon')
            plt.grid(True)
            plt.tight_layout()
            fname = os.path.join(outdir, f"{prefix}epsilon.png")
            plt.savefig(fname, dpi=150)
            plt.close()
        except Exception:
            pass

        # deliveries acumuladas
        try:
            plt.figure(figsize=(8,3))
            plt.plot(self.steps, self.deliveries)
            plt.xlabel('paso')
            plt.ylabel('entregas acumuladas')
            plt.title('entregas')
            plt.grid(True)
            plt.tight_layout()
            fname = os.path.join(outdir, f"{prefix}deliveries.png")
            plt.savefig(fname, dpi=150)
            plt.close()
        except Exception:
            pass

    def summary(self):
        """retorna un resumen rapido de las metricas en memoria"""
        return {
            'steps': len(self.steps),
            'reward_sum': float(sum(self.rewards)) if self.rewards else 0.0,
            'reward_mean': float(sum(self.rewards)/len(self.rewards)) if self.rewards else 0.0,
            'epsilon_last': float(self.eps[-1]) if self.eps else None,
            'deliveries_last': int(self.deliveries[-1]) if self.deliveries else 0,
        }

    def clear(self):
        self.steps.clear(); self.rewards.clear(); self.eps.clear(); self.deliveries.clear(); self.meta.clear()
