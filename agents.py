# agents.py
# definicion de agentes y cajas (ap.agent)
# editar: comportamiento de agentes, reward shaping, pickup/drop

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

    def step(self):
        bb = self.model.blackboard
        if self.task is None:
            self.task = bb.get_task(self.id)

        goal = None
        box_obj = None

        if self.task is not None:
            box_id = self.task["obj_id"]
            box_obj = next((b for b in self.model.boxes if b.id == box_id), None)
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

        def cell_occupied_by_agent(cell):
            for other in self.model.workers:
                if other is self: continue
                if getattr(other, "grid_pos", None) == cell:
                    return True
            return False

        prev_dist = None
        if goal is not None:
            prev_dist = abs(goal[0]-x) + abs(goal[1]-y)

        if action == "N":
            new_x = x
            new_y = clamp(y+1, 0, GRID-1)
            if (new_x, new_y) != (x, y) and not cell_occupied_by_agent((new_x, new_y)):
                moved = True
            else:
                reward += BLOCK_PENALTY

        elif action == "S":
            new_x = x
            new_y = clamp(y-1, 0, GRID-1)
            if (new_x, new_y) != (x, y) and not cell_occupied_by_agent((new_x, new_y)):
                moved = True
            else:
                reward += BLOCK_PENALTY

        elif action == "E":
            new_x = clamp(x+1, 0, GRID-1)
            new_y = y
            if (new_x, new_y) != (x, y) and not cell_occupied_by_agent((new_x, new_y)):
                moved = True
            else:
                reward += BLOCK_PENALTY

        elif action == "W":
            new_x = clamp(x-1, 0, GRID-1)
            new_y = y
            if (new_x, new_y) != (x, y) and not cell_occupied_by_agent((new_x, new_y)):
                moved = True
            else:
                reward += BLOCK_PENALTY

        elif action == "INTERACT":
            if (not self.carrying):
                local_box = next((b for b in self.model.boxes if b.grid_pos == (x,y)), None)
                if local_box is not None:
                    assigned_to = self.model.blackboard.assignments.get(local_box.id, None)
                    if assigned_to is None or assigned_to == self.id or (self.task is not None and self.task["obj_id"] == local_box.id):
                        self.carrying = True
                        self.carrying_box_id = local_box.id
                        self.model.blackboard.assignments[local_box.id] = self.id
                        reward += PICKUP_REWARD
                        print(f"[PICKUP] Agent{self.id} recogio Box{local_box.id} en {self.grid_pos}")
                        if self.task is None:
                            self.task = {"obj_id": local_box.id, "start": local_box.grid_pos, "target": DROP_ZONE, "done": False}
                    else:
                        reward += INTERACT_NOP_PENALTY
                else:
                    reward += INTERACT_NOP_PENALTY
            else:
                if self.carrying:
                    bid = self.carrying_box_id
                    if (x,y) == tuple(self.task["target"]) or (x,y) == DROP_ZONE or ( (x,y) and abs(x - DROP_ZONE[0])<=ZONE_RADIUS and abs(y - DROP_ZONE[1])<=ZONE_RADIUS ):
                        box = next((b for b in self.model.boxes if b.id == bid), None)
                        if box:
                            box.grid_pos = (x, y)
                            box.pos = grid_to_world(box.grid_pos)
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
                    else:
                        reward += INTERACT_NOP_PENALTY
                else:
                    reward += INTERACT_NOP_PENALTY

        elif action == "WAIT":
            reward += WAIT_PENALTY

        # fallback determinista si no hubo movimiento
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
                if (tx, ty) != (x, y) and not cell_occupied_by_agent((tx, ty)):
                    new_x, new_y = tx, ty
                    moved = True
                    reward += -0.05

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

        if self.battery <= 0.0:
            reward += BATTERY_ZERO_PENALTY
            if CHARGERS:
                self.grid_pos = CHARGERS[0]
                self.pos = grid_to_world(self.grid_pos)
            self.battery = 100.0
            self.carrying = False
            self.carrying_box_id = None
            self.task = None

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
