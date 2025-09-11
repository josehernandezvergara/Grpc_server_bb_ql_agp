# model.py
# definicion del modelo agentpy que instancia agentes, cajas y blackboard

import random
import agentpy as ap

from blackboard import BlackBoard
from agents import WorkerAgent, Box
from utils import grid_to_world
from settings import GRID, LOAD_ZONE, DROP_ZONE, ZONE_RADIUS, SAVE_EVERY, BOX_SPAWN_ZONES, AGENT_SPAWN_ZONES, OBSTACLES, WAIT_ZONE
from qlearning import qlearn

class WarehouseModel(ap.Model):
    def setup(self):
        # blackboard gestiona tareas y asignaciones
        self.blackboard = BlackBoard(self)

        # exponer obstaculos al modelo y permitir acceso desde agentes
        self.obstacles = set(OBSTACLES) if OBSTACLES is not None else set()

        # lista de agentes y cajas (agentpy)
        self.workers = ap.AgentList(self, self.p.agents, WorkerAgent)
        # crear dict id -> agente (se actualiza despues de crear workers)
        self.workers_dict = {ag.id: ag for ag in self.workers}
        self.boxes = ap.AgentList(self, self.p.objects, Box)

        # registro de pares de colision ya logueados (para evitar spam)
        self.collision_pairs_logged = set()

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
                # añadir tarea al blackboard
                self.blackboard.add_task(box.id, box.grid_pos, box.target)
        else:
            # fallback: colocacion aleatoria en grid
            for i, box in enumerate(self.boxes):
                rx = random.randrange(GRID)
                rz = random.randrange(GRID)
                box.grid_pos = (rx, rz)
                box.pos = grid_to_world(box.grid_pos)
                box.target = DROP_ZONE
                self.blackboard.add_task(box.id, box.grid_pos, box.target)

        # ajustar spawn de agentes: si AGENT_SPAWN_ZONES definido, reasignar sus posiciones
        if AGENT_SPAWN_ZONES:
            used = set()
            for i, w in enumerate(self.workers):
                zone = AGENT_SPAWN_ZONES[i % len(AGENT_SPAWN_ZONES)]
                gx = max(0, min(GRID-1, zone[0]))
                gz = max(0, min(GRID-1, zone[1]))
                # si la celda ya esta tomada, probar a desplazar un poco dentro del radius
                attempts = 0
                while (gx, gz) in used and attempts < 10:
                    gx = max(0, min(GRID-1, zone[0] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)))
                    gz = max(0, min(GRID-1, zone[1] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)))
                    attempts += 1
                w.grid_pos = (gx, gz)
                w.pos = grid_to_world(w.grid_pos)
                used.add((gx, gz))

        # counters y estadisticas
        self.step_counter = 0
        self.total_deliveries = 0

        # debug: imprimir configuracion y spawn para verificar
        print(f"[config] grid={GRID} drop_zone={DROP_ZONE} load_zone={LOAD_ZONE} zone_radius={ZONE_RADIUS}")
        for b in self.boxes:
            print(f"[spawn] Box{b.id} at {b.grid_pos} target={b.target}")
        for w in self.workers:
            print(f"[spawn] Agent{w.id} at {w.grid_pos}")

    def is_free_for_agent(self, cell, agent_id):
        """verifica si una celda esta libre para que un agente se mueva ahi"""
        if cell in self.obstacles:
            return False
        
        # verificar si hay otro agente en esa celda
        for other in self.workers:
            if other.id != agent_id and getattr(other, "grid_pos", None) == cell:
                return False
        
        return True
    
    def move_agent(self, agent, new_pos):
        """mueve un agente a una nueva posicion"""
        agent.grid_pos = new_pos
        agent.pos = grid_to_world(new_pos)

    def step(self):
        infos = []
        for ag in self.workers:
            info = ag.step()
            infos.append(info)
        self.step_counter += 1
        if self.step_counter % SAVE_EVERY == 0:
            qlearn.save(write_inference_snapshot=True)
