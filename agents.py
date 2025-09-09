# agents.py
# definicion de agentes y cajas (ap.agent)
# editar: comportamiento de agentes, reward shaping, pickup/drop
# comentarios en minuscula y sin acentos
#
# descripcion detallada:
# - clase WorkerAgent(ap.Agent):
#   - atributos de instancia creados en setup():
#     - grid_pos: tupla (gx,gz) posicion en la rejilla
#     - pos: lista [x,y,z] posicion en coordenadas world
#     - task: dict o None con keys {obj_id, start, target, done}
#     - carrying: bool si lleva caja
#     - carrying_box_id: id de la caja que lleva o None
#     - battery: float porcentaje bateria (0..100)
#     - total_steps: contador de pasos realizados
#     - roam_target: objetivo temporal cuando no hay tarea
#   - metodo step(): ejecuta logica por paso
#     - solicita tarea al blackboard si no tiene
#     - construye estado (x,y,carrying,batt_bin,rel_dx,rel_dy)
#     - selecciona accion via qlearn.choose_action
#     - maneja movimiento, interact, wait, fallback determinista
#     - actualiza bateria, aplica penalizaciones y recompensas
#     - actualiza qlearn con (s,a,r,s2) y decrece epsilon
#
# - clase Box(ap.Agent):
#   - atributos en setup(): grid_pos, pos, target
#   - carried_by: id del agente que la transporta (None si no)


import random
import numpy as np
import agentpy as ap

from settings import (
    GRID, CHARGERS, DROP_ZONE, WAIT_ZONE, ZONE_RADIUS,
    ACTIONS, MOVE_COST_BASE, CARRY_MULTIPLIER,
    PICKUP_REWARD, DROP_REWARD, BLOCK_PENALTY,
    WAIT_PENALTY, INTERACT_NOP_PENALTY, BATTERY_ZERO_PENALTY,
    DISTANCE_REWARD, DISTANCE_PENALTY, AGENT_SPAWN_ZONES
)
from settings import in_zone
from utils import grid_to_world, battery_to_bin, clamp
from qlearning import qlearn

class WorkerAgent(ap.Agent):
    def setup(self):
        # spawn preferente en agent spawn zones si el modelo las provee
        attempts = 0
        placed = False
        taken = { (w.grid_pos if hasattr(w,'grid_pos') else None) for w in getattr(self.model,'workers',[]) }

        spawn_zones = getattr(self.model, "agent_spawn_zones", None)
        if spawn_zones:
            for s in spawn_zones:
                if s not in taken:
                    self.grid_pos = s
                    placed = True
                    break

        while not placed and attempts < 100:
            rx = WAIT_ZONE[0] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)
            rz = WAIT_ZONE[1] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)
            rx = clamp(rx, 0, GRID-1)
            rz = clamp(rz, 0, GRID-1)
            if (rx, rz) not in taken:
                self.grid_pos = (rx, rz)
                placed = True
            attempts += 1
        if not placed:
            x = random.randrange(GRID)
            z = random.randrange(GRID)
            while (x,z) in taken:
                x = random.randrange(GRID)
                z = random.randrange(GRID)
            self.grid_pos = (x, z)

        self.pos = grid_to_world(self.grid_pos)
        self.task = None
        self.carrying = False
        self.carrying_box_id = None
        self.battery = 100.0
        self.total_steps = 0
        self.roam_target = None

    def step(self):
        bb = self.model.blackboard
        if self.task is None:
            self.task = bb.get_task(self.id)

        # determinar objetivo (goal)
        goal = None
        box_obj = None

        if self.task is not None:
            box_id = self.task["obj_id"]
            box_obj = next((b for b in self.model.boxes if b.id == box_id), None)
            if self.carrying:
                goal = DROP_ZONE
            else:
                if box_obj is not None and getattr(box_obj, "grid_pos", None) is not None:
                    goal = box_obj.grid_pos
                else:
                    goal = DROP_ZONE
        else:
            if not hasattr(self, "roam_target") or self.roam_target is None or random.random() < 0.05:
                wx = clamp(WAIT_ZONE[0] + random.randint(-2, 2), 0, GRID-1)
                wz = clamp(WAIT_ZONE[1] + random.randint(-2, 2), 0, GRID-1)
                self.roam_target = (wx, wz)
            goal = self.roam_target

        # features estado
        x, y = self.grid_pos
        batt_bin = battery_to_bin(self.battery)
        carrying_flag = 1 if self.carrying else 0
        if goal is None:
            rel_dx, rel_dy = 0, 0
        else:
            rel_dx = clamp(goal[0] - x, -3, 3)
            rel_dy = clamp(goal[1] - y, -3, 3)

        s_key = qlearn.state_key(x, y, carrying_flag, batt_bin, rel_dx, rel_dy)
        a = qlearn.choose_action(s_key)
        action = ACTIONS[a]

        reward = 0.0
        new_x, new_y = x, y
        moved = False

        def isObstacle(cell : tuple) -> bool:
            return cell in getattr(self.model, "obstacles", set())

        def cell_occupied_by_agent(cell):
            for other in self.model.workers:
                if other is self: continue
                if getattr(other, "grid_pos", None) == cell:
                    return True
            return False

        prev_dist = None
        if goal is not None:
            prev_dist = abs(goal[0]-x) + abs(goal[1]-y)

        # movimiento segun accion (N,S,E,W)
        if action == "N":
            new_x = x
            new_y = clamp(y+1, 0, GRID-1)
            if (new_x, new_y) != (x, y):
                if cell_occupied_by_agent((new_x, new_y)):
                    reward += BLOCK_PENALTY
                    print(f"[COLLISION] agent{self.id} blocked moving N from {(x,y)} by agent at {(new_x,new_y)}")
                elif isObstacle((new_x,new_y)):
                    reward += BLOCK_PENALTY
                    print(f"[COLLISION] agent{self.id} blocked moving N from {(x,y)} by obstacle at {(new_x,new_y)}")
                else:
                    moved = True
            else:
                reward += BLOCK_PENALTY

        elif action == "S":
            new_x = x
            new_y = clamp(y-1, 0, GRID-1)
            if (new_x, new_y) != (x, y):
                if cell_occupied_by_agent((new_x, new_y)):
                    reward += BLOCK_PENALTY
                    print(f"[COLLISION] agent{self.id} blocked moving S from {(x,y)} by agent at {(new_x,new_y)}")
                elif isObstacle((new_x,new_y)):
                    reward += BLOCK_PENALTY
                    print(f"[COLLISION] agent{self.id} blocked moving S from {(x,y)} by obstacle at {(new_x,new_y)}")
                else:
                    moved = True
            else:
                reward += BLOCK_PENALTY

        elif action == "E":
            new_x = clamp(x+1, 0, GRID-1)
            new_y = y
            if (new_x, new_y) != (x, y):
                if cell_occupied_by_agent((new_x, new_y)):
                    reward += BLOCK_PENALTY
                    print(f"[COLLISION] agent{self.id} blocked moving E from {(x,y)} by agent at {(new_x,new_y)}")
                elif isObstacle((new_x,new_y)):
                    reward += BLOCK_PENALTY
                    print(f"[COLLISION] agent{self.id} blocked moving E from {(x,y)} by obstacle at {(new_x,new_y)}")
                else:
                    moved = True
            else:
                reward += BLOCK_PENALTY

        elif action == "W":
            new_x = clamp(x-1, 0, GRID-1)
            new_y = y
            if (new_x, new_y) != (x, y):
                if cell_occupied_by_agent((new_x, new_y)):
                    reward += BLOCK_PENALTY
                    print(f"[COLLISION] agent{self.id} blocked moving W from {(x,y)} by agent at {(new_x,new_y)}")
                elif isObstacle((new_x,new_y)):
                    reward += BLOCK_PENALTY
                    print(f"[COLLISION] agent{self.id} blocked moving W from {(x,y)} by obstacle at {(new_x,new_y)}")
                else:
                    moved = True
            else:
                reward += BLOCK_PENALTY

        elif action == "INTERACT":
            # pickup oportunista mejorado: recoge caja si esta en la celda, incluso sin task asignada
            if (not self.carrying):
                local_box = next((b for b in self.model.boxes if getattr(b, "grid_pos", None) == (x,y)), None)
                if local_box is not None:
                    assigned_to = self.model.blackboard.assignments.get(local_box.id, None)
                    if assigned_to is None or assigned_to == self.id or (self.task is not None and self.task["obj_id"] == local_box.id):
                        # recoger
                        self.carrying = True
                        self.carrying_box_id = local_box.id
                        # marcar en el box que esta siendo llevado
                        local_box.carried_by = self.id
                        # quitar grid_pos para indicar que ya no esta en la celda fisica
                        local_box.grid_pos = None
                        # fijar pos de la caja a la posicion actual del agente
                        local_box.pos = self.pos[:] if hasattr(self.pos, "__iter__") else grid_to_world(self.grid_pos)
                        self.model.blackboard.assignments[local_box.id] = self.id
                        reward += PICKUP_REWARD
                        print(f"[PICKUP] agent{self.id} recogio box{local_box.id} en {x,y}")
                        if self.task is None:
                            self.task = {"obj_id": local_box.id, "start": (x,y), "target": DROP_ZONE, "done": False}
                    else:
                        reward += INTERACT_NOP_PENALTY
                else:
                    reward += INTERACT_NOP_PENALTY
            else:
                # drop: si esta en la zona de drop o en el target exacto
                if self.carrying:
                    bid = self.carrying_box_id
                    drop_ok = False
                    if in_zone((x,y), DROP_ZONE) or (self.task is not None and (x,y) == tuple(self.task["target"])):
                        drop_ok = True
                    if drop_ok:
                        box = next((b for b in self.model.boxes if b.id == bid), None)
                        if box:
                            box.carried_by = None
                            box.grid_pos = (x, y)
                            box.pos = grid_to_world(box.grid_pos)
                        self.model.blackboard.complete_task_for_box(bid)
                        self.carrying = False
                        self.carrying_box_id = None
                        self.roam_target = None  # volver a buscar wait
                        if self.task and self.task["obj_id"] == bid:
                            self.task = None
                        reward += DROP_REWARD
                        try:
                            self.model.total_deliveries += 1
                        except Exception:
                            pass
                        print(f"[DELIVERY] agent{self.id} entrego box{bid} en {(x,y)} (total={self.model.total_deliveries})")
                    else:
                        reward += INTERACT_NOP_PENALTY
                else:
                    reward += INTERACT_NOP_PENALTY

        elif action == "WAIT":
            reward += WAIT_PENALTY

        # fallback determinista: si la accion no movio y hay un goal, intentar acercarse en la direccion principal
        if (not moved) and (goal is not None):
            dx = goal[0] - x
            dy = goal[1] - y
            if dx != 0 or dy != 0:
                if abs(dx) > abs(dy):
                    intended = "E" if dx > 0 else "W"
                else:
                    intended = "N" if dy > 0 else "S"
                tx, ty = x, y
                if intended == "N":
                    ty = clamp(y+1, 0, GRID-1)
                elif intended == "S":
                    ty = clamp(y-1, 0, GRID-1)
                elif intended == "E":
                    tx = clamp(x+1, 0, GRID-1)
                elif intended == "W":
                    tx = clamp(x-1, 0, GRID-1)
                if (tx, ty) != (x, y) and not cell_occupied_by_agent((tx, ty)) and not isObstacle((tx, ty)):
                    new_x, new_y = tx, ty
                    moved = True
                    reward += -0.05

        # aplicar movimiento si se movio
        if moved:
            cost = MOVE_COST_BASE
            if self.carrying:
                cost *= CARRY_MULTIPLIER
            self.grid_pos = (new_x, new_y)
            self.pos = grid_to_world(self.grid_pos)
            self.battery = max(0.0, self.battery - cost)
            reward += -cost
            self.total_steps += 1
            if prev_dist is not None and goal is not None:
                new_dist = abs(goal[0]-new_x) + abs(goal[1]-new_y)
                if new_dist < prev_dist:
                    reward += DISTANCE_REWARD
                elif new_dist > prev_dist:
                    reward += DISTANCE_PENALTY

        # bateria a cero -> reiniciar y penalizar
        terminal = False
        if self.battery <= 0.0:
            reward += BATTERY_ZERO_PENALTY
            if CHARGERS:
                self.grid_pos = CHARGERS[0]
                self.pos = grid_to_world(self.grid_pos)
            self.battery = 100.0
            self.carrying = False
            self.carrying_box_id = None
            self.task = None
            terminal = True

        # estado siguiente y actualizacion q
        x2, y2 = self.grid_pos
        batt_bin2 = battery_to_bin(self.battery)
        carrying2 = 1 if self.carrying else 0
        if self.task is not None:
            box_obj2 = next((b for b in self.model.boxes if b.id == self.task["obj_id"]), None)
            if self.carrying:
                goal2 = tuple(self.task["target"])
            else:
                goal2 = box_obj2.grid_pos if box_obj2 is not None else None
        else:
            goal2 = None

        if goal2 is None:
            rel_dx2, rel_dy2 = 0, 0
        else:
            rel_dx2 = clamp(goal2[0] - x2, -3, 3)
            rel_dy2 = clamp(goal2[1] - y2, -3, 3)

        s2_key = qlearn.state_key(x2, y2, carrying2, batt_bin2, rel_dx2, rel_dy2)

        # actualizar q y epsilon
        qlearn.update(s_key, a, reward, s2_key)
        qlearn.decay_epsilon()

        return {
            "agent": self.id,
            "action": action,
            "reward": reward,
            "battery": self.battery,
            "pos": self.grid_pos
        }

class Box(ap.Agent):
    def setup(self):
        if not hasattr(self, "grid_pos") or self.grid_pos is None:
            x = random.randrange(GRID)
            z = random.randrange(GRID)
            self.grid_pos = (x, z)
        self.pos = grid_to_world(self.grid_pos)
        self.target = None
        # id del agente que la transporta (None si no la transporta)
        self.carried_by = None
