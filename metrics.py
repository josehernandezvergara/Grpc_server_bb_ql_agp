# metrics.py
from typing import List, Optional
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MetricsCollector:
    def __init__(self):
        self.steps: List[int] = []
        self.rewards: List[float] = []
        self.eps: List[float] = []
        self.deliveries: List[int] = []
        self.meta = {}
        self.trajectory = []  # lista de posiciones o estados del robot

    def record_position(self, position):
        """Registra la posición (o estado) del robot en la trayectoria."""
        self.trajectory.append(position)

    def save_trajectory_json(self, path: str):
        """Guarda la trayectoria en un archivo TXT en formato JSON."""
        import json
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.trajectory, f, indent=2, ensure_ascii=False)

    def record(self, step: int, reward: float, epsilon: float, deliveries: int):
        self.steps.append(int(step))
        self.rewards.append(float(reward))
        self.eps.append(float(epsilon))
        self.deliveries.append(int(deliveries))

    def save_csv(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'reward', 'epsilon', 'deliveries'])
            for s, r, e, d in zip(self.steps, self.rewards, self.eps, self.deliveries):
                w.writerow([s, r, e, d])

    def save_plots(self, outdir: str = '.', prefix: Optional[str] = ''):
        os.makedirs(outdir, exist_ok=True)

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
        return {
            'steps': len(self.steps),
            'reward_sum': float(sum(self.rewards)) if self.rewards else 0.0,
            'reward_mean': float(sum(self.rewards)/len(self.rewards)) if self.rewards else 0.0,
            'epsilon_last': float(self.eps[-1]) if self.eps else None,
            'deliveries_last': int(self.deliveries[-1]) if self.deliveries else 0,
        }

    def clear(self):
        self.steps.clear(); self.rewards.clear(); self.eps.clear(); self.deliveries.clear(); self.meta.clear(); self.trajectory.clear()
