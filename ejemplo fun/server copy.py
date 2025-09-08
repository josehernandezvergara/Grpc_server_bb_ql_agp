# server.py
# servidor grpc con simulacion agentpy + q-learning online en rejilla 12x12
# comentarios en minuscula y sin acentos

import random
import time
import os
import json
import pickle
import sys
import signal
import tempfile
from concurrent import futures

import grpc
import numpy as np
import agentpy as ap
import warehouse_pb2
import warehouse_pb2_grpc

# -----------------------
# configuracion por defecto (se escribe en q_config.json la primera vez)
# cambia aqui los valores que quieras ajustar facilmente
# -----------------------
CONFIG_PATH = "q_config.json"

default_config = {
    # q-learning hiperparametros basicos
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.3,              # exploracion inicial durante el entrenamiento
    "epsilon_decay": 0.9999,
    "min_epsilon": 0.05,
    # ambiente y guardado
    "grid_size": 12,
    "battery_bins": 11,
    "save_every_steps": 1000,
    "qfile": "qtable.pkl",                         # archivo principal de entrenamiento
    "inference_file": "qtable_0_inference.pkl",   # respaldo para modo inferencia (puedes renombrar si quieres)
    "snapshot_inference_on_save": True,            # si true, al guardar se escribe copia para inferencia
    "reset_on_train": False,                       # si true, al arrancar modo train se inicia q vacia aunque exista qfile
    # hardware/infra
    "charger_positions": [[0,0]],
    # acciones disponibles (orden fijo)
    "actions": ["N","S","E","W","INTERACT","WAIT"],
    # zonas (coordenadas de grid)
    "drop_zone": [11, 11],   # donde dejar cajas
    "wait_zone": [0, 0],     # donde esperar agentes sin tarea
    "load_zone": [0, 11],    # zona donde aparecen las cajas (spawn)
    "zone_radius": 1         # radio en celdas para considerar "dentro de la zona"
}

# crear q_config.json si no existe
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=2)
    config = default_config
else:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

# -----------------------
# constantes leidas desde el config
# -----------------------
ALPHA = config.get("alpha", 0.1)
GAMMA = config.get("gamma", 0.99)
EPSILON = config.get("epsilon", 0.3)
EPS_DECAY = config.get("epsilon_decay", 0.9999)
EPS_MIN = config.get("min_epsilon", 0.05)
GRID = config.get("grid_size", 12)
BATTERY_BINS = config.get("battery_bins", 11)
SAVE_EVERY = config.get("save_every_steps", 1000)
QFILE = config.get("qfile", "qtable.pkl")
INF_FILE = config.get("inference_file", "qtable_0_inference.pkl")
SNAPSHOT_INF_ON_SAVE = config.get("snapshot_inference_on_save", True)
RESET_ON_TRAIN = config.get("reset_on_train", False)
CHARGERS = [tuple(p) for p in config.get("charger_positions", [[0,0]])]
ACTIONS = config.get("actions", ["N","S","E","W","INTERACT","WAIT"])
NUM_ACTIONS = len(ACTIONS)

# zonas
DROP_ZONE = tuple(config.get("drop_zone", [GRID-1, GRID-1]))
WAIT_ZONE = tuple(config.get("wait_zone", [0, 0]))
LOAD_ZONE = tuple(config.get("load_zone", [0, GRID-1]))
ZONE_RADIUS = int(config.get("zone_radius", 1))

def in_zone(cell, zone, radius=ZONE_RADIUS):
    """retorna true si la celda 'cell' esta dentro de 'radius' de 'zone' (ambos tuplas (x,z))."""
    if cell is None or zone is None:
        return False
    dx = abs(int(cell[0]) - int(zone[0]))
    dz = abs(int(cell[1]) - int(zone[1]))
    return dx <= radius and dz <= radius

# -----------------------
# reward shaping (puedes ajustar para cambiar comportamiento)
# -----------------------
MOVE_COST_BASE = 0.3
CARRY_MULTIPLIER = 1.01
PICKUP_REWARD = 100.0
DROP_REWARD = 400.0
BLOCK_PENALTY = -5.0
WAIT_PENALTY = -0.2
INTERACT_NOP_PENALTY = -1.0
BATTERY_ZERO_PENALTY = -50.0
DISTANCE_REWARD = 1.0
DISTANCE_PENALTY = -0.5

# -----------------------
# utilidades de grid / conversion a mundo
# -----------------------
def clamp(v, a, b):
    return max(a, min(b, v))

def grid_to_world(cell):
    # conversion usada por unity: y fijo en 0.5
    x, y = cell
    return [float(x), 0.5, float(y)]

def world_to_grid(pos):
    x = int(round(pos[0]))
    z = int(round(pos[2]))
    x = clamp(x, 0, GRID-1)
    z = clamp(z, 0, GRID-1)
    return (x, z)

def battery_to_bin(batt):
    idx = int(round((batt/100.0) * (BATTERY_BINS-1)))
    return clamp(idx, 0, BATTERY_BINS-1)

# -----------------------
# modo de ejecucion: train o inference
# - preferimos usar la variable de entorno MODE
# - mantenemos compatibilidad con FORCE_INFERENCE (legacy)
# -----------------------
MODE = os.getenv("MODE", "").strip().lower()
if MODE == "":
    # compatibilidad con la variable anterior: FORCE_INFERENCE
    # si FORCE_INFERENCE esta establecida a "1" o "true" se interpreta como inference
    force_inf = os.getenv("FORCE_INFERENCE", "").strip().lower()
    if force_inf in ("1", "true", "yes"):
        MODE = "inference"
    else:
        MODE = "train"

# imprimir modo y configuracion basica
print(f"[CONFIG] modo={MODE} qfile={QFILE} inference_file={INF_FILE} reset_on_train={RESET_ON_TRAIN}")

# -----------------------
# q-learning manager mejorado con soporte de snapshot para inferencia
# - guarda y carga archivos de forma atomica
# - permite snapshot automatico para archivo de inferencia
# - respeta el modo: en inference epsilon=0 y no se actualiza la tabla
# -----------------------
class QLearning:
    def __init__(self, file_path=QFILE, inference_path=INF_FILE):
        # q: dict(state_key) -> np.array(num_actions)
        self.Q = {}
        self.epsilon = EPSILON
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.file = file_path
        self.inference_file = inference_path
        self.inference_mode = False

        # comportamiento segun el modo global
        if MODE == "inference":
            # intentar cargar snapshot de inferencia primero
            if os.path.exists(self.inference_file):
                self._load_file(self.inference_file)
                self.epsilon = 0.0
                self.decay_epsilon = lambda: None
                self.inference_mode = True
                print(f"[Q] cargada tabla de inferencia: {self.inference_file} entradas={len(self.Q)}")
            else:
                # fallback: intentar cargar qfile si existe (y convertir a inference)
                if os.path.exists(self.file):
                    self._load_file(self.file)
                    self.epsilon = 0.0
                    self.decay_epsilon = lambda: None
                    self.inference_mode = True
                    # crear snapshot inmediato si se desea
                    if SNAPSHOT_INF_ON_SAVE:
                        try:
                            self.save(write_inference_snapshot=True)
                        except Exception:
                            pass
                    print(f"[Q] inference-requested: inference snapshot no encontrada, se cargo qfile como fallback")
                else:
                    # no hay nada, tabla vacia pero en modo inference (no se explora)
                    self.Q = {}
                    self.epsilon = 0.0
                    self.decay_epsilon = lambda: None
                    self.inference_mode = True
                    print(f"[Q] modo inference y no se encontro archivo, tabla vacia iniciada (epsilon=0)")
        else:
            # modo training
            if RESET_ON_TRAIN:
                self.Q = {}
                print("[Q] modo train: reset_on_train activo -> tabla vacia iniciada")
            else:
                if os.path.exists(self.file):
                    self._load_file(self.file)
                else:
                    print(f"[Q] modo train: no se encontro {self.file}, iniciar con tabla vacia")

    # cargar desde ruta especifica
    def _load_file(self, path):
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                # coercion de llaves a tuplas por seguridad
                self.Q = {tuple(k): np.array(v, dtype=np.float32) for k, v in loaded.items()}
                print(f"[Q] cargada {path} entradas={len(self.Q)}")
            else:
                print(f"[Q] formato inesperado en {path}, ignorando")
        except Exception as e:
            print(f"[Q] error cargando {path}: {e}")

    # asegurar que la key existe en Q y devolver el vector
    def _ensure(self, key):
        if key not in self.Q:
            self.Q[key] = np.zeros(NUM_ACTIONS, dtype=np.float32)
        return self.Q[key]

    def state_key(self, x, y, carrying, batt_bin, rel_dx, rel_dy):
        return (int(x), int(y), int(carrying), int(batt_bin), int(rel_dx), int(rel_dy))

    def choose_action(self, state_key):
        if random.random() < self.epsilon:
            return random.randrange(NUM_ACTIONS)
        qvals = self._ensure(state_key)
        maxv = np.max(qvals)
        choices = np.flatnonzero(qvals == maxv)
        return int(np.random.choice(choices))

    def update(self, s_key, a, r, s2_key):
        # si estamos en modo inference, no actualizamos la tabla
        if self.inference_mode:
            return
        q_s = self._ensure(s_key)
        q_s2 = self._ensure(s2_key)
        qsa = q_s[a]
        qmax_next = np.max(q_s2)
        q_s[a] = qsa + self.alpha * (r + self.gamma * qmax_next - qsa)

    def decay_epsilon(self):
        # metodo por defecto (puede sobrescribirse en inference)
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    # guarda la qtable en self.file y opcionalmente escribe tambien inference_file
    # usa escritura atomica: escribe en tempfile y os.replace
    def save(self, write_inference_snapshot=False):
        try:
            serial = {k: v.tolist() for k, v in self.Q.items()}
            # guardar archivo principal de forma atomica
            dirpath = os.path.dirname(os.path.abspath(self.file)) or "."
            with tempfile.NamedTemporaryFile("wb", delete=False, dir=dirpath) as tf:
                pickle.dump(serial, tf)
                tmpname = tf.name
            os.replace(tmpname, self.file)
            print(f"[Q] q-table guardada en {self.file} entries={len(self.Q)}")
            # snapshot para inferencia si se solicita o si config lo indica
            if write_inference_snapshot or SNAPSHOT_INF_ON_SAVE:
                dirinf = os.path.dirname(os.path.abspath(self.inference_file)) or "."
                with tempfile.NamedTemporaryFile("wb", delete=False, dir=dirinf) as tif:
                    pickle.dump(serial, tif)
                    tmpinf = tif.name
                os.replace(tmpinf, self.inference_file)
                print(f"[Q] snapshot para inferencia guardado en {self.inference_file}")
        except Exception as e:
            print(f"[Q] error guardando q-table: {e}")

    # metodo util para forzar cargar la tabla de inferencia (si existe)
    def load_inference(self):
        if os.path.exists(self.inference_file):
            self._load_file(self.inference_file)
            self.epsilon = 0.0
            self.decay_epsilon = lambda: None
            self.inference_mode = True
            print("[MODE] inferencia: epsilon fijado a 0.0 desde inference_file")
            return True
        return False

# instancia global de qlearn
qlearn = QLearning(file_path=QFILE, inference_path=INF_FILE)

# si arrancamos en modo inference, aseguramos que la tabla de inferencia este cargada
if MODE == "inference" and (not qlearn.inference_mode):
    # intento extra de cargar snapshot si no se cargo en el constructor
    loaded_inf = qlearn.load_inference()
    if not loaded_inf and os.path.exists(QFILE):
        # si no existe inference_file pero existe qfile: cargar qfile y crear snapshot
        print("[MODE] inference solicitado pero snapshot no encontrado; cargando qfile y creando snapshot")
        qlearn._load_file(QFILE)
        qlearn.epsilon = 0.0
        qlearn.decay_epsilon = lambda: None
        qlearn.inference_mode = True
        try:
            qlearn.save(write_inference_snapshot=True)
        except Exception:
            pass

# ------------------------
# blackboard para tareas (gestiona lista de tareas y asignaciones)
# ------------------------
class BlackBoard:
    def __init__(self, model):
        self.model = model
        self.tasks = []        # lista de dicts {obj_id, start, target, done}
        self.assignments = {}  # map box_id -> agent_id

    def add_task(self, obj_id, start_grid, target_grid, min_dist=0.5):
        self.tasks.append({
            "obj_id": obj_id,
            "start": start_grid,
            "target": target_grid,
            "done": False
        })

    def get_task(self, agent_id):
        # devuelve la primera tarea no hecha y sin asignar
        for task in self.tasks:
            if not task["done"] and task["obj_id"] not in self.assignments:
                self.assignments[task["obj_id"]] = agent_id
                return task
        return None

    def complete_task_for_box(self, box_id):
        for task in self.tasks:
            if task["obj_id"] == box_id:
                task["done"] = True
                if box_id in self.assignments:
                    del self.assignments[box_id]
                return True
        # si no existe la tarea, limpiar assignment si hay
        if box_id in self.assignments:
            del self.assignments[box_id]
        return False

    def assign_task_to_agent(self, agent_id, box_id, target_grid):
        # asigna task existente o crea una nueva
        for task in self.tasks:
            if task["obj_id"] == box_id:
                task["target"] = target_grid
                task["done"] = False
                self.assignments[box_id] = agent_id
                return True
        box = next((b for b in self.model.boxes if b.id == box_id), None)
        start = box.grid_pos if box is not None else (random.randrange(GRID), random.randrange(GRID))
        self.add_task(box_id, start, target_grid)
        self.assignments[box_id] = agent_id
        return True

# ------------------------
# agente trabajador (worker)
# - spawn inicial cerca de la zona wait
# - comportamiento:
#   - si tiene tarea -> mover a la caja -> interact para pickup -> llevar a drop_zone -> drop
#   - si no tiene tarea -> vagabundear cerca de wait_zone (roam target)
# - fallback determinista para acercarse al goal si la politica q no mueve
# ------------------------
class WorkerAgent(ap.Agent):
    def setup(self):
        # spawn en la zona de espera (wait zone) para evitar discrepancias
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
                # cuando esta cargando, objetivo = drop zone
                goal = DROP_ZONE
            else:
                # si existe la caja, objetivo = posicion caja
                if box_obj is not None:
                    goal = box_obj.grid_pos
                else:
                    # fallback
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

        def cell_occupied_by_agent(cell):
            for other in self.model.workers:
                if other is self: continue
                if getattr(other, "grid_pos", None) == cell:
                    return True
            return False

        # distancia previa para shaping
        prev_dist = None
        if goal is not None:
            prev_dist = abs(goal[0]-x) + abs(goal[1]-y)

        # movimiento segun accion (N,S,E,W)
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
            # pickup oportunista mejorado: recoge caja si esta en la celda, incluso sin task asignada
            if (not self.carrying):
                local_box = next((b for b in self.model.boxes if b.grid_pos == (x,y)), None)
                if local_box is not None:
                    assigned_to = self.model.blackboard.assignments.get(local_box.id, None)
                    # si la caja esta libre o asignada a mi o es la de mi task: recoger
                    if assigned_to is None or assigned_to == self.id or (self.task is not None and self.task["obj_id"] == local_box.id):
                        self.carrying = True
                        self.carrying_box_id = local_box.id
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
                    if in_zone((x,y), DROP_ZONE) or (self.task is not None and (x,y) == tuple(self.task["target"])):
                        box = next((b for b in self.model.boxes if b.id == bid), None)
                        if box:
                            # guardar caja en la celda actual (o en DROP_ZONE)
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

        # ---------
        # fallback determinista: si la accion no movio y hay un goal, intentar acercarse en la direccion principal
        # esto hace que las zonas se respeten aunque la q-table sugiera wait o acciones bloqueadas
        # ---------
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
                    # small negative reward for forced move to keep learning stable
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

# ------------------------
# clase box: su posicion la decide warehousemodel al spawnear
# ------------------------
class Box(ap.Agent):
    def setup(self):
        # si warehousemodel no define posicion, usamos aleatoria por seguridad
        if not hasattr(self, "grid_pos") or self.grid_pos is None:
            x = random.randrange(GRID)
            z = random.randrange(GRID)
            self.grid_pos = (x, z)
        self.pos = grid_to_world(self.grid_pos)
        self.target = None

# ------------------------
# modelo del warehouse
# - spawnea agentes cerca de wait_zone
# - spawnea cajas alrededor de load_zone
# - crea tareas para cada caja (start -> load pos, target -> drop zone)
# ------------------------
class WarehouseModel(ap.Model):
    def setup(self):
        self.blackboard = BlackBoard(self)
        self.workers = ap.AgentList(self, self.p.agents, WorkerAgent)
        self.workers_dict = {ag.id: ag for ag in self.workers}
        self.boxes = ap.AgentList(self, self.p.objects, Box)

        # spawn de cajas centrado en load_zone (dentro de zone_radius)
        for i, box in enumerate(self.boxes):
            rx = LOAD_ZONE[0] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)
            rz = LOAD_ZONE[1] + random.randint(-ZONE_RADIUS, ZONE_RADIUS)
            rx = clamp(rx, 0, GRID-1)
            rz = clamp(rz, 0, GRID-1)
            box.grid_pos = (rx, rz)
            box.pos = grid_to_world(box.grid_pos)
            # target por defecto = drop zone
            box.target = DROP_ZONE
            # añadimos la tarea al blackboard
            self.blackboard.add_task(box.id, box.grid_pos, box.target)

        self.step_counter = 0
        self.total_deliveries = 0

        # debug: imprimir configuracion y spawn para verificar
        print(f"[CONFIG] grid={GRID} drop_zone={DROP_ZONE} load_zone={LOAD_ZONE} wait_zone={WAIT_ZONE} zone_radius={ZONE_RADIUS}")
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

# ------------------------
# implementacion grpc: metodos expuestos al cliente unity
# - getcoords: devuelve timestamp y lista de objects con id/position/speed
# - pickup/drop/assigntask: endpoints para controlar el blackboard/estado
# ------------------------
class WarehouseService(warehouse_pb2_grpc.WarehouseServiceServicer):
    def __init__(self, model):
        self.model = model

    def GetCoords(self, request, context):
        # cada llamada avanza 1 step del modelo
        self.model.step()
        objects = []
        for ag in self.model.workers:
            obj = warehouse_pb2.ObjectData(
                id=f"Agent{ag.id}",
                position=warehouse_pb2.Position(x=ag.pos[0], y=ag.pos[1], z=ag.pos[2]),
                speed=1.0
            )
            objects.append(obj)
        for box in self.model.boxes:
            obj = warehouse_pb2.ObjectData(
                id=f"Box{box.id}",
                position=warehouse_pb2.Position(x=box.pos[0], y=box.pos[1], z=box.pos[2]),
                speed=0.5
            )
            objects.append(obj)
        return warehouse_pb2.CoordsResponse(timestamp=int(time.time()), objects=objects)

    def Pickup(self, request, context):
        aid = int(request.agent.replace("Agent", ""))
        bid = int(request.box.replace("Box", ""))
        ag = self.model.workers_dict.get(aid)
        if ag:
            ag.carrying = True
            ag.carrying_box_id = bid
            # asignar en el blackboard
            self.model.blackboard.assignments[bid] = aid
            # actualizar caja en server para reflejar pick up (opcional: referencia visual)
            box = next((b for b in self.model.boxes if b.id == bid), None)
            if box:
                box.grid_pos = ag.grid_pos
                box.pos = grid_to_world(box.grid_pos)
        return warehouse_pb2.Ack(ok=True)

    def Drop(self, request, context):
        aid = int(request.agent.replace("Agent", ""))
        bid = int(request.box.replace("Box", ""))
        pos = request.position
        box = next((b for b in self.model.boxes if b.id == bid), None)
        if box:
            gx, gz = world_to_grid([pos.x, pos.y, pos.z])
            box.grid_pos = (gx, gz)
            box.pos = grid_to_world(box.grid_pos)
        self.model.blackboard.complete_task_for_box(bid)
        ag = self.model.workers_dict.get(aid)
        if ag:
            ag.carrying = False
            ag.carrying_box_id = None
        return warehouse_pb2.Ack(ok=True)

    def AssignTask(self, request, context):
        aid = int(request.agent.replace("Agent", ""))
        bid = int(request.box.replace("Box", ""))
        tx = int(round(request.target.x))
        tz = int(round(request.target.z))
        if aid not in self.model.workers_dict:
            return warehouse_pb2.Ack(ok=False, error="Agent not found")
        box = next((b for b in self.model.boxes if b.id == bid), None)
        if box is None:
            return warehouse_pb2.Ack(ok=False, error="Box not found")
        target = (clamp(tx,0,GRID-1), clamp(tz,0,GRID-1))
        self.model.blackboard.assign_task_to_agent(aid, bid, target)
        return warehouse_pb2.Ack(ok=True)

# ------------------------
# manejador de señales para guardar q y cerrar servidor limpio
# ------------------------
GRPC_SERVER = None

def handle_sigterm(signum, frame):
    print("signal recibido, guardando q-table y saliendo...")
    try:
        qlearn.save(write_inference_snapshot=True)
    except Exception as e:
        print("error guardando Q:", e)
    try:
        if GRPC_SERVER is not None:
            GRPC_SERVER.stop(0)
    except Exception:
        pass
    sys.exit(0)

# ------------------------
# main: crea modelo y levanta servidor grpc
# ------------------------
if __name__ == "__main__":
    # parametros: ajusta cuantos agentes y cajas quieres
    parameters = {"agents": 3, "objects": 2, "steps": 1000}
    model = WarehouseModel(parameters)
    model.setup()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    warehouse_pb2_grpc.add_WarehouseServiceServicer_to_server(WarehouseService(model), server)

    GRPC_SERVER = server
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    print("servidor grpc escuchando en 0.0.0.0:50051...")
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("deteniendo servidor...")
        qlearn.save(write_inference_snapshot=True)
        server.stop(0)
