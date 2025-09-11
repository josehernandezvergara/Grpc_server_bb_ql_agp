# model.py
# definicion del modelo agentpy que instancia agentes, cajas y blackboard
# editar: numeros de agentes/objetos, spawn y logica del ciclo
# comentarios en minuscula y sin acentos
#
# descripcion:
# - WarehouseModel(ap.Model)
#   - atributos:
#     - blackboard: instancia de BlackBoard que contiene tareas y asignaciones
#     - obstacles: set de tuplas (gx,gz) con celdas bloqueadas
#     - box_spawn_zones, agent_spawn_zones: listas de tuplas para spawn
#     - workers: AgentList de WorkerAgent
#     - workers_dict: mapa id->agente
#     - boxes: AgentList de Box
#     - step_counter, total_deliveries: contadores de simulacion
#   - metodos:
#     - setup(): crea blackboard, obstaculos, spawns, agentes y cajas
#     - step(): llama ag.step() para cada agente y guarda periodicamente Q-table


import random
import agentpy as ap

from blackboard import BlackBoard
from agents import WorkerAgent, Box
from utils import grid_to_world
from settings import GRID, LOAD_ZONE, DROP_ZONE, ZONE_RADIUS, SAVE_EVERY, BOX_SPAWN_ZONES, AGENT_SPAWN_ZONES, OBSTACLES
from qlearning import qlearn

class WarehouseModel(ap.Model):
    def setup(self):
        # blackboard gestiona tareas y asignaciones
        self.blackboard = BlackBoard(self)

        # exponer obstaculos al modelo y permitir acceso desde agentes
        self.obstacles = set(OBSTACLES) if OBSTACLES is not None else set()

        # lista de agentes y cajas (agentpy)
        self.workers = ap.AgentList(self, self.p.agents, WorkerAgent)
        self.workers_dict = {ag.id: ag for ag in self.workers}
        self.boxes = ap.AgentList(self, self.p.objects, Box)

        # registro de pares de colision ya logueados (para evitar spam)
        self.collision_pairs_logged = set()

        # spawn de cajas: si hay zonas de spawn en world definidas (BOX_SPAWN_ZONES),
        # usamos esas coordenadas convertidas a grid (en settings ya se normalizo BOX_SPAWN_ZONES)
        if BOX_SPAWN_ZONES:
            for i, box in enumerate(self.boxes):
                # usar un spawn zone de la lista, ciclando si hay menos zonas que cajas
                zone = BOX_SPAWN_ZONES[i % len(BOX_SPAWN_ZONES)]
                # agregar un pequeño offset aleatorio dentro del zone_radius para diversidad
                rx = clamp = None
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
                # asignar posiciones secuenciales de la lista (ciclando si es necesario)
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
            # guardar qtable y snapshot de inferencia si esta configurado
            qlearn.save(write_inference_snapshot=True)