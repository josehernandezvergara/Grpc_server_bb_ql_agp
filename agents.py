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

from settings import (GRID, CHARGERS, DROP_ZONE, WAIT_ZONE, ZONE_RADIUS,
                      ACTIONS, MOVE_COST_BASE, CARRY_MULTIPLIER,
                      PICKUP_REWARD, DROP_REWARD, BLOCK_PENALTY,
                      WAIT_PENALTY, INTERACT_NOP_PENALTY, BATTERY_ZERO_PENALTY,
                      DISTANCE_REWARD, DISTANCE_PENALTY)
from utils import grid_to_world, battery_to_bin, clamp
from qlearning import qlearn

class WorkerAgent(ap.Agent):
    def setup(self):
        # spawn inicial: intentamos ubicar cerca de wait_zone para evitar discrepancias
        attempts = 0
        placed = False
        taken = { (w.grid_pos if hasattr(w,'grid_pos') else None) for w in getattr(self.model,'workers',[]) }
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
            # fallback a posicion aleatoria si no pudo ubicar cerca de wait
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

    def _occupying_agent(self, cell):
        """retorna el agente que ocupa la celda o None"""
        for other in self.model.workers:
            if other is self: continue
            if getattr(other, "grid_pos", None) == cell:
                return other
        return None

    def step(self):
        bb = self.model.blackboard
        # si no tengo tarea, intentar obtener una
        if self.task is None:
            self.task = bb.get_task(self.id)

        # determinar objetivo (goal)
        goal = None
        box_obj = None

        if self.task is not None:
            box_id = self.task["obj_id"]
            # buscar la caja correspondiente
            box_obj = next((b for b in self.model.boxes if b.id == box_id), None)

            # si la caja ya no existe o fue recogida por otro, cancelar la tarea
            if (not self.carrying) and box_obj is None:
                # evitar que vayamos a drop sin motivo: cancelar tarea y liberar assignment
                print(f"[TASK] Agent{self.id} cancela tarea Box{box_id}: caja no encontrada o ya recogida")
                bb.complete_task_for_box(box_id)
                self.task = None
                # despues de cancelar, definimos roam_target para volver a wait
                self.roam_target = (clamp(WAIT_ZONE[0] + random.randint(-1,1), 0, GRID-1),
                                    clamp(WAIT_ZONE[1] + random.randint(-1,1), 0, GRID-1))
                goal = self.roam_target
            else:
                # si estoy cargando -> objetivo drop
                if self.carrying:
                    goal = DROP_ZONE
                else:
                    # si existe la caja, objetivo = posicion caja
                    if box_obj is not None:
                        goal = box_obj.grid_pos
                    else:
                        goal = DROP_ZONE
        else:
            # sin tarea: moverse alrededor de wait_zone (roam)
            if not hasattr(self, "roam_target") or self.roam_target is None or random.random() < 0.05:
                wx = clamp(WAIT_ZONE[0] + random.randint(-2, 2), 0, GRID-1)
                wz = clamp(WAIT_ZONE[1] + random.randint(-2, 2), 0, GRID-1)
                self.roam_target = (wx, wz)
            goal = self.roam_target

        # features de estado para la q
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

        # distancia previa para shaping
        prev_dist = None
        if goal is not None:
            prev_dist = abs(goal[0]-x) + abs(goal[1]-y)

        # movimiento segun accion (N,S,E,W)
        if action == "N":
            new_x = x
            new_y = clamp(y+1, 0, GRID-1)
            occupant = self._occupying_agent((new_x, new_y))
            if (new_x, new_y) != (x, y) and occupant is None and (new_x, new_y) not in getattr(self.model, "obstacles", set()):
                moved = True
            else:
                reward += BLOCK_PENALTY
                # log de colision una sola vez por par
                if occupant is not None:
                    pair = tuple(sorted((self.id, occupant.id)))
                    if pair not in self.model.collision_pairs_logged:
                        print(f"[COLLISION] Agent{self.id} intento mover a {(new_x,new_y)} pero Agent{occupant.id} ocupa la celda")
                        self.model.collision_pairs_logged.add(pair)

        elif action == "S":
            new_x = x
            new_y = clamp(y-1, 0, GRID-1)
            occupant = self._occupying_agent((new_x, new_y))
            if (new_x, new_y) != (x, y) and occupant is None and (new_x, new_y) not in getattr(self.model, "obstacles", set()):
                moved = True
            else:
                reward += BLOCK_PENALTY
                if occupant is not None:
                    pair = tuple(sorted((self.id, occupant.id)))
                    if pair not in self.model.collision_pairs_logged:
                        print(f"[COLLISION] Agent{self.id} intento mover a {(new_x,new_y)} pero Agent{occupant.id} ocupa la celda")
                        self.model.collision_pairs_logged.add(pair)

        elif action == "E":
            new_x = clamp(x+1, 0, GRID-1)
            new_y = y
            occupant = self._occupying_agent((new_x, new_y))
            if (new_x, new_y) != (x, y) and occupant is None and (new_x, new_y) not in getattr(self.model, "obstacles", set()):
                moved = True
            else:
                reward += BLOCK_PENALTY
                if occupant is not None:
                    pair = tuple(sorted((self.id, occupant.id)))
                    if pair not in self.model.collision_pairs_logged:
                        print(f"[COLLISION] Agent{self.id} intento mover a {(new_x,new_y)} pero Agent{occupant.id} ocupa la celda")
                        self.model.collision_pairs_logged.add(pair)

        elif action == "W":
            new_x = clamp(x-1, 0, GRID-1)
            new_y = y
            occupant = self._occupying_agent((new_x, new_y))
            if (new_x, new_y) != (x, y) and occupant is None and (new_x, new_y) not in getattr(self.model, "obstacles", set()):
                moved = True
            else:
                reward += BLOCK_PENALTY
                if occupant is not None:
                    pair = tuple(sorted((self.id, occupant.id)))
                    if pair not in self.model.collision_pairs_logged:
                        print(f"[COLLISION] Agent{self.id} intento mover a {(new_x,new_y)} pero Agent{occupant.id} ocupa la celda")
                        self.model.collision_pairs_logged.add(pair)

        elif action == "INTERACT":
            # pickup oportunista mejorado: recoge caja si esta en la celda, incluso sin task asignada
            if (not self.carrying):
                local_box = next((b for b in self.model.boxes if getattr(b,"grid_pos",None) == (x,y)), None)
                if local_box is not None:
                    assigned_to = self.model.blackboard.assignments.get(local_box.id, None)
                    # si la caja esta libre o asignada a mi o es la de mi task: recoger
                    if assigned_to is None or assigned_to == self.id or (self.task is not None and self.task["obj_id"] == local_box.id):
                        self.carrying = True
                        self.carrying_box_id = local_box.id
                        # marcar en el box que esta siendo llevado
                        local_box.carried_by = self.id
                        local_box.grid_pos = None
                        local_box.pos = self.pos[:]  # posicion actual world
                        self.model.blackboard.assignments[local_box.id] = self.id
                        reward += PICKUP_REWARD
                        print(f"[PICKUP] Agent{self.id} recogio Box{local_box.id} en {self.grid_pos}")
                        # si no tenia task, creamos una interna para seguimiento
                        if self.task is None:
                            self.task = {"obj_id": local_box.id, "start": local_box.grid_pos, "target": DROP_ZONE, "done": False}
                    else:
                        reward += INTERACT_NOP_PENALTY
                else:
                    reward += INTERACT_NOP_PENALTY
            else:
                # drop: si esta en la zona de drop o en el target exacto
                if self.carrying:
                    bid = self.carrying_box_id
                    # comprobar si estamos dentro de drop zone
                    in_drop = (abs(x-DROP_ZONE[0]) <= ZONE_RADIUS and abs(y-DROP_ZONE[1]) <= ZONE_RADIUS)
                    target_match = (self.task is not None and (x,y) == tuple(self.task["target"]))
                    if in_drop or target_match:
                        box = next((b for b in self.model.boxes if b.id == bid), None)
                        if box:
                            box.grid_pos = (x, y)
                            box.pos = grid_to_world(box.grid_pos)
                            box.carried_by = None
                        self.model.blackboard.complete_task_for_box(bid)
                        # reset carrier state
                        self.carrying = False
                        self.carrying_box_id = None
                        if self.task and self.task["obj_id"] == bid:
                            self.task = None
                        reward += DROP_REWARD
                        try:
                            self.model.total_deliveries += 1
                        except Exception:
                            pass
                        print(f"[DELIVERY] Agent{self.id} entrego Box{bid} en {(x,y)} (total={self.model.total_deliveries})")
                        # despues de drop, mover roam_target hacia wait para que el agente no quede en drop
                        self.roam_target = (clamp(WAIT_ZONE[0] + random.randint(-1,1), 0, GRID-1),
                                            clamp(WAIT_ZONE[1] + random.randint(-1,1), 0, GRID-1))
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
                occ = self._occupying_agent((tx, ty))
                if (tx, ty) != (x, y) and occ is None and (tx,ty) not in getattr(self.model, "obstacles", set()):
                    new_x, new_y = tx, ty
                    moved = True
                    reward += -0.05
                else:
                    # si no pude mover por un agent distinto, logueo once por par
                    if occ is not None:
                        pair = tuple(sorted((self.id, occ.id)))
                        if pair not in self.model.collision_pairs_logged:
                            print(f"[COLLISION] Agent{self.id} fallback intento mover a {(tx,ty)} pero Agent{occ.id} ocupa la celda")
                            self.model.collision_pairs_logged.add(pair)

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
            # si cargo una caja, hacer que la caja siga al agente
            if self.carrying and self.carrying_box_id is not None:
                box = next((b for b in self.model.boxes if b.id == self.carrying_box_id), None)
                if box:
                    box.pos = self.pos[:]
                    box.grid_pos = None
                    box.carried_by = self.id
            # shaping por distancia al objetivo
            if prev_dist is not None and goal is not None:
                new_dist = abs(goal[0]-new_x) + abs(goal[1]-new_y)
                if new_dist < prev_dist:
                    reward += DISTANCE_REWARD
                elif new_dist > prev_dist:
                    reward += DISTANCE_PENALTY

        # bateria a cero -> reiniciar y penalizar
        if self.battery <= 0.0:
            reward += BATTERY_ZERO_PENALTY
            if CHARGERS:
                self.grid_pos = CHARGERS[0]
                self.pos = grid_to_world(self.grid_pos)
            self.battery = 100.0
            self.carrying = False
            self.carrying_box_id = None
            self.task = None

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

        # actualizar q y epsilon (qlearn internamente ignora updates en inference mode)
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
        self.carried_by = None