# agents.py
# definicion de agentes y cajas (ap.agent)
# comentarios en minuscula y sin acentos

import random
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
        # spawn inicial cerca de wait_zone
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
        """retorna agente que ocupa la celda o None"""
        for other in self.model.workers:
            if other is self: continue
            if getattr(other, "grid_pos", None) == cell:
                return other
        return None

    def step(self):
        bb = self.model.blackboard
        if self.task is None:
            self.task = bb.get_task(self.id)

        goal = None
        box_obj = None

        if self.task is not None:
            box_id = self.task["obj_id"]
            box_obj = next((b for b in self.model.boxes if b.id == box_id), None)

            # si la caja ya no existe o fue recogida por otro, cancelar tarea
            if (not self.carrying) and box_obj is None:
                print(f"[TASK] Agent{self.id} cancela tarea Box{box_id}: caja no encontrada o ya recogida")
                bb.complete_task_for_box(box_id)
                self.task = None
                self.roam_target = (clamp(WAIT_ZONE[0] + random.randint(-1,1), 0, GRID-1),
                                    clamp(WAIT_ZONE[1] + random.randint(-1,1), 0, GRID-1))
                goal = self.roam_target
            else:
                if self.carrying:
                    goal = DROP_ZONE
                else:
                    if box_obj is not None:
                        goal = box_obj.grid_pos
                    else:
                        goal = DROP_ZONE
        else:
            if not hasattr(self, "roam_target") or self.roam_target is None or random.random() < 0.05:
                wx = clamp(WAIT_ZONE[0] + random.randint(-2, 2), 0, GRID-1)
                wz = clamp(WAIT_ZONE[1] + random.randint(-2, 2), 0, GRID-1)
                self.roam_target = (wx, wz)
            goal = self.roam_target

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

        prev_dist = None
        if goal is not None:
            prev_dist = abs(goal[0]-x) + abs(goal[1]-y)

        # movimiento N/S/E/W con comprobaciones de obstaculos y agentes
        if action == "N":
            new_x = x
            new_y = clamp(y+1, 0, GRID-1)
            occupant = self._occupying_agent((new_x, new_y))
            if (new_x, new_y) != (x, y) and occupant is None and (new_x, new_y) not in getattr(self.model, "obstacles", set()):
                moved = True
            else:
                reward += BLOCK_PENALTY
                if occupant is not None:
                    pair = tuple(sorted((self.id, occupant.id)))
                    key = ('aa', pair[0], pair[1])
                    if key not in self.model.collision_monitor:
                        print(f"[COLLISION] Agent{self.id} intento mover a {(new_x,new_y)} pero Agent{occupant.id} ocupa la celda")
                        self.model.collision_monitor[key] = self.model.collision_ttl_default
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
                    key = ('aa', pair[0], pair[1])
                    if key not in self.model.collision_monitor:
                        print(f"[COLLISION] Agent{self.id} intento mover a {(new_x,new_y)} pero Agent{occupant.id} ocupa la celda")
                        self.model.collision_monitor[key] = self.model.collision_ttl_default
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
                    key = ('aa', pair[0], pair[1])
                    if key not in self.model.collision_monitor:
                        print(f"[COLLISION] Agent{self.id} intento mover a {(new_x,new_y)} pero Agent{occupant.id} ocupa la celda")
                        self.model.collision_monitor[key] = self.model.collision_ttl_default
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
                    key = ('aa', pair[0], pair[1])
                    if key not in self.model.collision_monitor:
                        print(f"[COLLISION] Agent{self.id} intento mover a {(new_x,new_y)} pero Agent{occupant.id} ocupa la celda")
                        self.model.collision_monitor[key] = self.model.collision_ttl_default
                        self.model.collision_pairs_logged.add(pair)

        elif action == "INTERACT":
            # pickup oportunista: guardar start antes de cambiar box.grid_pos a None
            if (not self.carrying):
                local_box = next((b for b in self.model.boxes if getattr(b,"grid_pos",None) == (x,y)), None)
                if local_box is not None:
                    assigned_to = self.model.blackboard.assignments.get(local_box.id, None)
                    if assigned_to is None or assigned_to == self.id or (self.task is not None and self.task["obj_id"] == local_box.id):
                        # guardar start antes de quitar grid_pos
                        start_pos = local_box.grid_pos
                        self.carrying = True
                        self.carrying_box_id = local_box.id
                        local_box.carried_by = self.id
                        local_box.grid_pos = None
                        local_box.pos = self.pos[:]  # posicion actual world
                        self.model.blackboard.assignments[local_box.id] = self.id
                        reward += PICKUP_REWARD
                        print(f"[PICKUP] Agent{self.id} recogio Box{local_box.id} en {self.grid_pos}")
                        if self.task is None:
                            self.task = {"obj_id": local_box.id, "start": start_pos, "target": DROP_ZONE, "done": False}
                    else:
                        reward += INTERACT_NOP_PENALTY
                else:
                    reward += INTERACT_NOP_PENALTY
            else:
                if self.carrying:
                    bid = self.carrying_box_id
                    in_drop = (abs(x-DROP_ZONE[0]) <= ZONE_RADIUS and abs(y-DROP_ZONE[1]) <= ZONE_RADIUS)
                    target_match = (self.task is not None and (x,y) == tuple(self.task["target"]))
                    if in_drop or target_match:
                        box = next((b for b in self.model.boxes if b.id == bid), None)
                        if box:
                            box.grid_pos = (x, y)
                            box.pos = grid_to_world(box.grid_pos)
                            box.carried_by = None
                        self.model.blackboard.complete_task_for_box(bid)
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
                        self.roam_target = (clamp(WAIT_ZONE[0] + random.randint(-1,1), 0, GRID-1),
                                            clamp(WAIT_ZONE[1] + random.randint(-1,1), 0, GRID-1))
                    else:
                        reward += INTERACT_NOP_PENALTY
                else:
                    reward += INTERACT_NOP_PENALTY

        elif action == "WAIT":
            reward += WAIT_PENALTY

        # fallback determinista
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
                    if occ is not None:
                        pair = tuple(sorted((self.id, occ.id)))
                        key = ('aa', pair[0], pair[1])
                        if key not in self.model.collision_monitor:
                            print(f"[COLLISION] Agent{self.id} fallback intento mover a {(tx,ty)} pero Agent{occ.id} ocupa la celda")
                            self.model.collision_monitor[key] = self.model.collision_ttl_default
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
            if self.carrying and self.carrying_box_id is not None:
                box = next((b for b in self.model.boxes if b.id == self.carrying_box_id), None)
                if box:
                    box.pos = self.pos[:]
                    box.grid_pos = None
                    box.carried_by = self.id
            if prev_dist is not None and goal is not None:
                new_dist = abs(goal[0]-new_x) + abs(goal[1]-new_y)
                if new_dist < prev_dist:
                    reward += DISTANCE_REWARD
                elif new_dist > prev_dist:
                    reward += DISTANCE_PENALTY

        # bateria a cero
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
