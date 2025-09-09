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
from settings import GRID, LOAD_ZONE, DROP_ZONE, ZONE_RADIUS, SAVE_EVERY, BOX_SPAWN_ZONES, AGENT_SPAWN_ZONES, OBSTACLES, WAIT_ZONE
from qlearning import qlearn

class WarehouseModel(ap.Model):
    def setup(self):
        self.blackboard = BlackBoard(self)

        # inicializar obstaculos: si vienen desde settings los usamos tal cual,
        # si no vienen, generamos 3 obstaculos "centrales" como puntos medios
        # entre wait, load y drop para cubrir los 3 cuadrantes.
        self.obstacles = set(OBSTACLES) if OBSTACLES else set()

        if not self.obstacles:
            # zonas base para generar puntos medios
            zones = [WAIT_ZONE, LOAD_ZONE, DROP_ZONE]
            mids = set()
            for i in range(len(zones)):
                for j in range(i+1, len(zones)):
                    a = zones[i]
                    b = zones[j]
                    mx = (a[0] + b[0]) // 2
                    mz = (a[1] + b[1]) // 2
                    # clamp por si acaso
                    mx = max(0, min(GRID-1, mx))
                    mz = max(0, min(GRID-1, mz))
                    mids.add((mx, mz))
            self.obstacles = mids
            # imprimir una sola vez el mapeo obstacles (grid->world)
            mapped = [grid_to_world(o) for o in sorted(self.obstacles)]
            print(f"[OBSTACLES] generados automaticamente (grid)={sorted(self.obstacles)} (world)={mapped}")
        else:
            # si vienen definidos, imprimir mapeo una sola vez
            mapped = []
            for o in sorted(self.obstacles):
                try:
                    mapped.append(grid_to_world(o))
                except Exception:
                    mapped.append(o)
            print(f"[OBSTACLES] cargados desde config (grid)={sorted(self.obstacles)} (world)={mapped}")

        # exponer spawn zones procesadas desde settings
        self.box_spawn_zones = BOX_SPAWN_ZONES
        self.agent_spawn_zones = AGENT_SPAWN_ZONES

        # crear listas de agentes y cajas
        self.workers = ap.AgentList(self, self.p.agents, WorkerAgent)
        self.workers_dict = {ag.id: ag for ag in self.workers}
        self.boxes = ap.AgentList(self, self.p.objects, Box)

        # spawn de cajas centrado en box_spawn_zones si estan definidas
        for i, box in enumerate(self.boxes):
            if self.box_spawn_zones:
                g = self.box_spawn_zones[i % len(self.box_spawn_zones)]
                rx, rz = g
            else:
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
            qlearn.save(write_inference_snapshot=True)
