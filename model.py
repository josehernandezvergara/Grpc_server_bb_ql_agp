# model.py
# definicion del modelo agentpy que instancia agentes, cajas y blackboard
# editar: numeros de agentes/objetos, spawn y logica del ciclo

import random
import agentpy as ap

from blackboard import BlackBoard
from agents import WorkerAgent, Box
from utils import grid_to_world
from settings import GRID, LOAD_ZONE, DROP_ZONE, ZONE_RADIUS, SAVE_EVERY, OBSTACLES
from qlearning import qlearn

class WarehouseModel(ap.Model):
    def setup(self):
        self.blackboard = BlackBoard(self)
        self.obstacles = OBSTACLES
        self.workers = ap.AgentList(self, self.p.agents, WorkerAgent)
        self.workers_dict = {ag.id: ag for ag in self.workers}
        self.boxes = ap.AgentList(self, self.p.objects, Box)

        for i, box in enumerate(self.boxes):
            rx = LOAD_ZONE[0] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)
            rz = LOAD_ZONE[1] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)
            rx = max(0, min(GRID-1, rx))
            rz = max(0, min(GRID-1, rz))
            box.grid_pos = (rx, rz)
            box.pos = grid_to_world(box.grid_pos)
            box.target = DROP_ZONE
            self.blackboard.add_task(box.id, box.grid_pos, box.target)

        self.step_counter = 0
        self.total_deliveries = 0

        print(f"[CONFIG] grid={GRID} drop_zone={DROP_ZONE} load_zone={LOAD_ZONE} zone_radius={ZONE_RADIUS}")
        for b in self.boxes:
            print(f"[SPAWN] Box{b.id} at {b.grid_pos} target={b.target}")
        for w in self.workers:
            print(f"[SPAWN] Agent{w.id} at {w.grid_pos}")

    def step(self):
        infos = []
        for ag in self.workers:
            info = ag.step()
            infos.append(info)
        self.step_counter += 1
        if self.step_counter % SAVE_EVERY == 0:
            qlearn.save(write_inference_snapshot=True)
