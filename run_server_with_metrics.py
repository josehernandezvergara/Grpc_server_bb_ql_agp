# run_server_with_metrics.py
# script para ejecutar servidor + recoleccion de metricas sin tocar codigo principal

import os
import signal
import sys
import time
from concurrent import futures

import grpc

from metrics import MetricsCollector
import settings
from model import WarehouseModel
from grpc_service import WarehouseService
import warehouse_pb2_grpc
from qlearning import qlearn
from agents import WorkerAgent

GRPC_SERVER = None
SAVE_DIR = 'metrics'
os.makedirs(SAVE_DIR, exist_ok=True)
MC = MetricsCollector()

def instrument_classes():
    orig_agent_step = WorkerAgent.step

    def wrapped_agent_step(self, *args, **kwargs):
        info = orig_agent_step(self, *args, **kwargs)
        try:
            if not hasattr(self.model, '_last_agent_infos'):
                self.model._last_agent_infos = []
            self.model._last_agent_infos.append(info)
        except Exception:
            pass
        return info

    WorkerAgent.step = wrapped_agent_step

    from model import WarehouseModel as WM
    orig_model_step = WM.step

    def wrapped_model_step(self, *args, **kwargs):
        try:
            self._last_agent_infos = []
        except Exception:
            pass
        result = orig_model_step(self, *args, **kwargs)
        try:
            infos = getattr(self, '_last_agent_infos', []) or []
            step_reward = 0.0
            for info in infos:
                if isinstance(info, dict):
                    step_reward += float(info.get('reward', 0.0))
            MC.record(step=getattr(self, 'step_counter', 0), reward=step_reward, epsilon=qlearn.epsilon, deliveries=getattr(self, 'total_deliveries', 0))
        except Exception:
            pass
        return result

    WM.step = wrapped_model_step

def handle_sig(signum, frame):
    print('\nsignal recibido, guardando metricas y q-table...')
    try:
        MC.save_csv(os.path.join(SAVE_DIR, 'metrics_on_signal.csv'))
        MC.save_plots(SAVE_DIR, prefix='signal_')
        MC.save_trajectory_json(os.path.join(SAVE_DIR, 'trajectory_on_signal.txt'))
    except Exception as e:
        print('error guardando metricas:', e)
    try:
        qlearn.save(write_inference_snapshot=True)
    except Exception as e:
        print('error guardando q-table:', e)
    try:
        if GRPC_SERVER is not None:
            GRPC_SERVER.stop(0)
    except Exception:
        pass
    sys.exit(0)

def main():
    mode = settings.MODE
    print(f'[RUN_METRICS] settings.MODE={mode}')

    instrument_classes()

    parameters = {"agents": 3, "objects": 2, "steps": 0}
    model = WarehouseModel(parameters)
    model.setup()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    warehouse_pb2_grpc.add_WarehouseServiceServicer_to_server(WarehouseService(model), server)

    global GRPC_SERVER
    GRPC_SERVER = server

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    print(f"servidor grpc (instrumentado) escuchando en 0.0.0.0:50051... mode={mode}")
    server.add_insecure_port("0.0.0.0:50051")
    server.start()

    try:
        while True:
            time.sleep(5)
            try:
                if len(MC.steps) > 0:
                    MC.save_csv(os.path.join(SAVE_DIR, 'metrics_latest.csv'))
            except Exception:
                pass
    except KeyboardInterrupt:
        handle_sig(None, None)

if __name__ == '__main__':
    main()
