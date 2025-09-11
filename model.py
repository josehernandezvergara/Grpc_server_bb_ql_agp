# model.py
# definicion del modelo agentpy que instancia agentes, cajas y blackboard
# comentarios en minuscula y sin acentos

import random
import agentpy as ap

from blackboard import BlackBoard
from agents import WorkerAgent, Box
from utils import grid_to_world
from settings import GRID, LOAD_ZONE, DROP_ZONE, ZONE_RADIUS, SAVE_EVERY, BOX_SPAWN_ZONES, AGENT_SPAWN_ZONES, OBSTACLES, TRAIN_QUADRANT
from qlearning import qlearn

class WarehouseModel(ap.Model):
    def setup(self):
        # blackboard gestiona tareas y asignaciones
        self.blackboard = BlackBoard(self)

        # exponer obstaculos al modelo y permitir acceso desde agentes
        self.obstacles = set(OBSTACLES) if OBSTACLES is not None else set()

        # lista de agentes y cajas (agentpy)
        self.workers = ap.AgentList(self, self.p.agents, WorkerAgent)
        # workers_dict inicial (se actualiza luego si es necesario)
        self.workers_dict = {ag.id: ag for ag in self.workers}
        self.boxes = ap.AgentList(self, self.p.objects, Box)

        # collision monitor: map key -> ttl y set para compatibilidad
        self.collision_monitor = {}
        self.collision_ttl_default = 50
        self.collision_pairs_logged = set()

        # aplicar train_quadrant si se definio y estamos en modo train
        # spawn forzado dentro del cuadrante (si existe)
        if TRAIN_QUADRANT is not None and getattr(self.p, 'mode', None) != 'inference':
            xmin, ymin, xmax, ymax = TRAIN_QUADRANT
            for i, box in enumerate(self.boxes):
                rx = random.randint(xmin, xmax)
                rz = random.randint(ymin, ymax)
                box.grid_pos = (rx, rz)
                box.pos = grid_to_world(box.grid_pos)
                box.target = DROP_ZONE
                self.blackboard.add_task(box.id, box.grid_pos, box.target)
        else:
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
        if AGENT_SPAWN_ZONES:
            used = set()
            for i, w in enumerate(self.workers):
                zone = AGENT_SPAWN_ZONES[i % len(AGENT_SPAWN_ZONES)]
                gx = max(0, min(GRID-1, zone[0]))
                gz = max(0, min(GRID-1, zone[1]))
                attempts = 0
                while (gx, gz) in used and attempts < 10:
                    gx = max(0, min(GRID-1, zone[0] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)))
                    gz = max(0, min(GRID-1, zone[1] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)))
                    attempts += 1
                w.grid_pos = (gx, gz)
                w.pos = grid_to_world(w.grid_pos)
                used.add((gx, gz))
        else:
            # fallback: aleatorio
            used = set()
            for w in self.workers:
                rx = random.randrange(GRID)
                rz = random.randrange(GRID)
                while (rx, rz) in used:
                    rx = random.randrange(GRID)
                    rz = random.randrange(GRID)
                w.grid_pos = (rx, rz)
                w.pos = grid_to_world(w.grid_pos)
                used.add((rx, rz))

        # reconstruir workers_dict por si cambian ids/orden
        self.workers_dict = {ag.id: ag for ag in self.workers}

        # counters y estadisticas
        self.step_counter = 0
        self.total_deliveries = 0

        # debug: imprimir configuracion y spawn (mantener formato original)
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

        # decrementar ttl del collision_monitor
        to_remove = []
        for k in list(self.collision_monitor.keys()):
            self.collision_monitor[k] -= 1
            if self.collision_monitor[k] <= 0:
                to_remove.append(k)
        for k in to_remove:
            del self.collision_monitor[k]

        if self.step_counter % SAVE_EVERY == 0:
            qlearn.save(write_inference_snapshot=True)
