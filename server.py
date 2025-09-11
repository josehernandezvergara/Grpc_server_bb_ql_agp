# server.py
import signal
import sys
import time
from concurrent import futures

import grpc

from model import WarehouseModel
from grpc_service import WarehouseService
import warehouse_pb2_grpc
from settings import MODE
from qlearning import qlearn

GRPC_SERVER = None

def handle_sigterm(signum, frame):
    print("signal recibido, guardando q-table y saliendo...")
    try:
        qlearn.save(write_inference_snapshot=True)
    except Exception as e:
        print("error guardando Q:", e)
    try:
        if GRPC_SERVER is not None:
            GRPC_SERVER.stop(0)
    except Exception:
        pass
    sys.exit(0)

if __name__ == "__main__":
    parameters = {"agents": 3, "objects": 2, "steps": 1000}
    model = WarehouseModel(parameters)
    model.setup()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    warehouse_pb2_grpc.add_WarehouseServiceServicer_to_server(WarehouseService(model), server)

    GRPC_SERVER = server
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    print(f"servidor grpc escuchando en 0.0.0.0:50051... modo={MODE}")
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("deteniendo servidor...")
        qlearn.save(write_inference_snapshot=True)
        server.stop(0)
