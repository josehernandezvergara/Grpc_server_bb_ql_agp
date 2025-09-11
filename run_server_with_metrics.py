"""
run_server_with_metrics.py

arranca el servidor grpc en modo similar a server.py pero incorpora metricas
sin modificar archivos existentes. el script:
- instrumenta WorkerAgent.step para almacenar los infos en model._last_agent_infos
- instrumenta WarehouseModel.step para, despues de ejecutar la logica original,
  agregar un registro en MetricsCollector con reward total del paso, epsilon y entregas
- guarda las metricas periodicamente y al recibir señal SIGINT/SIGTERM

uso:
  set MODE=train   # opcional
  python run_server_with_metrics.py

nota: este script es independiente y no modifica server.py ni otros modulos.
"""

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
# asegurar que la carpeta de metricas exista antes de crear el collector
os.makedirs(SAVE_DIR, exist_ok=True)
MC = MetricsCollector()


def instrument_classes():
    """monkeypatch WorkerAgent.step y WarehouseModel.step para recolectar metricas."""
    # wrap WorkerAgent.step to append returned info to model._last_agent_infos
    orig_agent_step = WorkerAgent.step

    def wrapped_agent_step(self, *args, **kwargs):
        info = orig_agent_step(self, *args, **kwargs)
        try:
            if not hasattr(self.model, '_last_agent_infos'):
                self.model._last_agent_infos = []
            # info puede ser None o dict
            self.model._last_agent_infos.append(info)
        except Exception:
            pass
        return info

    WorkerAgent.step = wrapped_agent_step

    # wrap WarehouseModel.step to call original and then record aggregated metrics
    from model import WarehouseModel as WM
    orig_model_step = WM.step

    def wrapped_model_step(self, *args, **kwargs):
        # reset buffer
        try:
            self._last_agent_infos = []
        except Exception:
            pass
        # call original step (this will call WorkerAgent.step which fills _last_agent_infos)
        result = orig_model_step(self, *args, **kwargs)
        # aggregate reward
        try:
            infos = getattr(self, '_last_agent_infos', []) or []
            step_reward = 0.0
            for info in infos:
                if isinstance(info, dict):
                    step_reward += float(info.get('reward', 0.0))
            # record metrics: use current step_counter as step id
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
    # permitir forzar modo train si se desea
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

    # loop principal: guardar metricas cada tanto
    try:
        while True:
            time.sleep(5)
            # guardar snapshot periodico ligero
            try:
                if len(MC.steps) > 0:
                    MC.save_csv(os.path.join(SAVE_DIR, 'metrics_latest.csv'))
            except Exception:
                pass
    except KeyboardInterrupt:
        handle_sig(None, None)


if __name__ == '__main__':
    main()
