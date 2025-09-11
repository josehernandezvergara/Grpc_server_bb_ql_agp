# model.py
import random
import agentpy as ap
import time

from blackboard import BlackBoard
from agents import WorkerAgent, Box
from utils import grid_to_world, clamp
from settings import GRID, LOAD_ZONE, DROP_ZONE, ZONE_RADIUS, SAVE_EVERY, BOX_SPAWN_ZONES, AGENT_SPAWN_ZONES, OBSTACLES, TRAIN_QUADRANT, MODE
from qlearning import qlearn

class WarehouseModel(ap.Model):
    def setup(self):
        # blackboard gestiona tareas y asignaciones
        self.blackboard = BlackBoard(self)

        # obstaculos expuestos
        self.obstacles = set(OBSTACLES) if OBSTACLES is not None else set()

        # listas agentpy
        self.workers = ap.AgentList(self, self.p.agents, WorkerAgent)
        self.boxes = ap.AgentList(self, self.p.objects, Box)

        # spawn de cajas
        if BOX_SPAWN_ZONES:
            for i, box in enumerate(self.boxes):
                zone = BOX_SPAWN_ZONES[i % len(BOX_SPAWN_ZONES)]
                rx = zone[0] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)
                rz = zone[1] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)
                rx = max(0, min(GRID-1, rx))
                rz = max(0, min(GRID-1, rz))
                box.grid_pos = (rx, rz)
                box.pos = grid_to_world(box.grid_pos)
                box.target = DROP_ZONE
                self.blackboard.add_task(box.id, box.grid_pos, box.target)
        else:
            for i, box in enumerate(self.boxes):
                rx = random.randrange(GRID)
                rz = random.randrange(GRID)
                box.grid_pos = (rx, rz)
                box.pos = grid_to_world(box.grid_pos)
                box.target = DROP_ZONE
                self.blackboard.add_task(box.id, box.grid_pos, box.target)

        # spawn de agentes
        used = set()
        if AGENT_SPAWN_ZONES:
            for i, w in enumerate(self.workers):
                zone = AGENT_SPAWN_ZONES[i % len(AGENT_SPAWN_ZONES)]
                gx = max(0, min(GRID-1, zone[0]))
                gz = max(0, min(GRID-1, zone[1]))
                attempts = 0
                while (gx, gz) in used and attempts < 20:
                    gx = max(0, min(GRID-1, zone[0] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)))
                    gz = max(0, min(GRID-1, zone[1] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)))
                    attempts += 1
                w.grid_pos = (gx, gz)
                w.pos = grid_to_world(w.grid_pos)
                used.add((gx, gz))
        else:
            # random spawn
            used = set()
            for w in self.workers:
                attempts = 0
                placed = False
                while not placed and attempts < 200:
                    x = random.randrange(GRID)
                    z = random.randrange(GRID)
                    if (x,z) not in used:
                        w.grid_pos = (x,z)
                        w.pos = grid_to_world(w.grid_pos)
                        used.add((x,z))
                        placed = True
                    attempts += 1
                if not placed:
                    w.grid_pos = (0,0)
                    w.pos = grid_to_world(w.grid_pos)

        # si estamos en modo train y TRAIN_QUADRANT definido, forzar spawn dentro del cuadrante
        if MODE == "train" and TRAIN_QUADRANT:
            xmin, ymin, xmax, ymax = TRAIN_QUADRANT
            # cajas
            for box in self.boxes:
                box.grid_pos = (random.randint(xmin, xmax), random.randint(ymin, ymax))
                box.pos = grid_to_world(box.grid_pos)
                # actualizar tarea si existe
                # busca y actualiza en blackboard
            # agents
            for w in self.workers:
                w.grid_pos = (random.randint(xmin, xmax), random.randint(ymin, ymax))
                w.pos = grid_to_world(w.grid_pos)

        # actualizar diccionario id->agente
        self.workers_dict = {ag.id: ag for ag in self.workers}

        # counters y estadisticas
        self.step_counter = 0
        self.total_deliveries = 0

        # monitor de colisiones para evitar spam
        self.collision_pairs_logged = set()
        self.collision_monitor = {}  # map key -> ttl
        self.collision_ttl_default = 50

        # debug: imprimir configuracion y spawn
        print(f"[CONFIG] grid={GRID} drop_zone={DROP_ZONE} load_zone={LOAD_ZONE} zone_radius={ZONE_RADIUS}")
        for b in self.boxes:
            print(f"[spawn] Box{b.id} at {b.grid_pos} target={b.target}")
        for w in self.workers:
            print(f"[spawn] Agent{w.id} at {w.grid_pos}")

    def step(self):
        infos = []
        for ag in self.workers:
            info = ag.step()
            infos.append(info)
        self.step_counter += 1

        # decrementar ttl del collision_monitor (limpieza)
        to_remove = []
        for k in list(self.collision_monitor.keys()):
            self.collision_monitor[k] -= 1
            if self.collision_monitor[k] <= 0:
                to_remove.append(k)
        for k in to_remove:
            del self.collision_monitor[k]

        if self.step_counter % SAVE_EVERY == 0:
            try:
                qlearn.save(write_inference_snapshot=True)
            except Exception:
                pass

        return infos
