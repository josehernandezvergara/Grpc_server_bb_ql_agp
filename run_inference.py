# run_inference.py
import pickle
import server

# cargar qtable (si existe)
with open("qtable.pkl", "rb") as f:
    server.qlearn.Q = {tuple(k): server.np.array(v, dtype=server.np.float32) for k,v in pickle.load(f).items()}

server.qlearn.epsilon = 0.0  # explotación pura

params = {"agents": 3, "objects": 5, "steps": 0}
model = server.WarehouseModel(params)
model.setup()

# correr N pasos y contar entregas
for i in range(2000):
    for ag in model.workers:
        ag.step()

print("Entregas en inference:", model.total_deliveries)
