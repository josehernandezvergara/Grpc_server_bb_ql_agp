# settings.py
# carga de configuracion y constantes compartidas
# comentarios en minuscula y sin acentos

import os
import json

CONFIG_PATH = "q_config.json"

default_config = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.3,
    "epsilon_decay": 0.9999,
    "min_epsilon": 0.05,
    "grid_size": 12,
    "cell_size": 1.0,
    "origin": [0, 0],
    "battery_bins": 11,
    "save_every_steps": 1000,
    "qfile": "qtable.pkl",
    "inference_file": "qtable_0_inference.pkl",
    "snapshot_inference_on_save": True,
    "reset_on_train": False,
    "charger_positions": [[0,0]],
    "actions": ["N","S","E","W","INTERACT","WAIT"],
    "drop_zone": [11, 11],
    "wait_zone": [0, 0],
    "load_zone": [0, 11],
    "zone_radius": 1,
    "box_spawn_zones_world": [],
    "agent_spawn_zones_world": [],
    "delivery_zone_world": None,
    "obstacles": [],
    "move_cost_base": 0.3,
    "carry_multiplier": 1.01,
    "pickup_reward": 100.0,
    "drop_reward": 400.0,
    "block_penalty": -5.0,
    "wait_penalty": -0.2,
    "interact_nop_penalty": -1.0,
    "battery_zero_penalty": -50.0,
    "distance_reward": 1.0,
    "distance_penalty": -0.5,
    # opcional: si present in config, entrenar en un cuadrante
    # "train_quadrant": [xmin, ymin, xmax, ymax]
}

# crear q_config.json si no existe (se escribe valores por defecto)
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=2)
    config = default_config
else:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

def _clamp(v, a, b):
    return max(a, min(b, v))

# q-learning
ALPHA = config.get("alpha", default_config["alpha"])
GAMMA = config.get("gamma", default_config["gamma"])
EPSILON = config.get("epsilon", default_config["epsilon"])
EPS_DECAY = config.get("epsilon_decay", default_config["epsilon_decay"])
EPS_MIN = config.get("min_epsilon", default_config["min_epsilon"])

GRID = int(config.get("grid_size", default_config["grid_size"]))
CELL_SIZE = float(config.get("cell_size", default_config["cell_size"]))
ORIGIN = tuple(config.get("origin", default_config["origin"]))

BATTERY_BINS = int(config.get("battery_bins", default_config["battery_bins"]))
SAVE_EVERY = int(config.get("save_every_steps", default_config["save_every_steps"]))
QFILE = config.get("qfile", default_config["qfile"])
INF_FILE = config.get("inference_file", default_config["inference_file"])
SNAPSHOT_INF_ON_SAVE = config.get("snapshot_inference_on_save", default_config["snapshot_inference_on_save"])
RESET_ON_TRAIN = config.get("reset_on_train", default_config["reset_on_train"])
CHARGERS = [tuple(p) for p in config.get("charger_positions", default_config["charger_positions"])]
ACTIONS = config.get("actions", default_config["actions"])
NUM_ACTIONS = len(ACTIONS)

DROP_ZONE = tuple(config.get("drop_zone", default_config["drop_zone"]))
WAIT_ZONE = tuple(config.get("wait_zone", default_config["wait_zone"]))
LOAD_ZONE = tuple(config.get("load_zone", default_config["load_zone"]))
ZONE_RADIUS = int(config.get("zone_radius", default_config["zone_radius"]))

_BOX_SPAWN_WORLD = config.get("box_spawn_zones_world", default_config.get("box_spawn_zones_world", []))
_AGENT_SPAWN_WORLD = config.get("agent_spawn_zones_world", default_config.get("agent_spawn_zones_world", []))
_DELIVERY_WORLD = config.get("delivery_zone_world", default_config.get("delivery_zone_world", None))

def _world_to_grid_tuple(p):
    """convierte lista [x,y,z] world a tupla grid (gx,gz) usando origin y cell_size"""
    if p is None or len(p) < 3:
        return None
    gx = int(round((float(p[0]) / CELL_SIZE) + ORIGIN[0]))
    gz = int(round((float(p[2]) / CELL_SIZE) + ORIGIN[1]))
    gx = _clamp(gx, 0, GRID-1)
    gz = _clamp(gz, 0, GRID-1)
    return (gx, gz)

BOX_SPAWN_ZONES = []
for _p in _BOX_SPAWN_WORLD:
    g = _world_to_grid_tuple(_p)
    if g is not None:
        BOX_SPAWN_ZONES.append(g)

AGENT_SPAWN_ZONES = []
for _p in _AGENT_SPAWN_WORLD:
    g = _world_to_grid_tuple(_p)
    if g is not None:
        AGENT_SPAWN_ZONES.append(g)

DELIVERY_ZONE = _world_to_grid_tuple(_DELIVERY_WORLD) if _DELIVERY_WORLD is not None else None

# obstacles: pueden venir en grid [gx,gz] o en world [x,y,z]
_OBS_RAW = config.get("obstacles", default_config["obstacles"])
OBSTACLES = set()
for p in _OBS_RAW:
    if isinstance(p, (list, tuple)) and len(p) == 2:
        gx = int(p[0]); gz = int(p[1])
        gx = _clamp(gx, 0, GRID-1); gz = _clamp(gz, 0, GRID-1)
        OBSTACLES.add((gx, gz))
    elif isinstance(p, (list, tuple)) and len(p) >= 3:
        g = _world_to_grid_tuple(p)
        if g is not None:
            OBSTACLES.add(g)

# optional train_quadrant
TRAIN_QUADRANT_RAW = config.get("train_quadrant", None)
TRAIN_QUADRANT = None
if TRAIN_QUADRANT_RAW and isinstance(TRAIN_QUADRANT_RAW, (list, tuple)) and len(TRAIN_QUADRANT_RAW) == 4:
    xmin = max(0, min(GRID-1, int(TRAIN_QUADRANT_RAW[0])))
    ymin = max(0, min(GRID-1, int(TRAIN_QUADRANT_RAW[1])))
    xmax = max(0, min(GRID-1, int(TRAIN_QUADRANT_RAW[2])))
    ymax = max(0, min(GRID-1, int(TRAIN_QUADRANT_RAW[3])))
    TRAIN_QUADRANT = (xmin, ymin, xmax, ymax)

# reward shaping
MOVE_COST_BASE = float(config.get("move_cost_base", default_config["move_cost_base"]))
CARRY_MULTIPLIER = float(config.get("carry_multiplier", default_config["carry_multiplier"]))
PICKUP_REWARD = float(config.get("pickup_reward", default_config["pickup_reward"]))
DROP_REWARD = float(config.get("drop_reward", default_config["drop_reward"]))
BLOCK_PENALTY = float(config.get("block_penalty", default_config["block_penalty"]))
WAIT_PENALTY = float(config.get("wait_penalty", default_config["wait_penalty"]))
INTERACT_NOP_PENALTY = float(config.get("interact_nop_penalty", default_config["interact_nop_penalty"]))
BATTERY_ZERO_PENALTY = float(config.get("battery_zero_penalty", default_config["battery_zero_penalty"]))
DISTANCE_REWARD = float(config.get("distance_reward", default_config["distance_reward"]))
DISTANCE_PENALTY = float(config.get("distance_penalty", default_config["distance_penalty"]))

# modo: env MODE o FORCE_INFERENCE
MODE = os.getenv("MODE", "").strip().lower()
if MODE == "":
    force_inf = os.getenv("FORCE_INFERENCE", "").strip().lower()
    if force_inf in ("1","true","yes"):
        MODE = "inference"
    else:
        MODE = "train"

def in_zone(cell, zone, radius=ZONE_RADIUS):
    """retorna true si la celda cell esta en radius de zone"""
    if cell is None or zone is None:
        return False
    dx = abs(int(cell[0]) - int(zone[0]))
    dz = abs(int(cell[1]) - int(zone[1]))
    return dx <= radius and dz <= radius

# impresiones de diagnostico (mantener formato de logs existente por compatibilidad)
print(f"[SETTINGS] mode={MODE} grid={GRID} cell_size={CELL_SIZE} origin={ORIGIN}")
print(f"[SETTINGS] box_spawn_zones(grid)={BOX_SPAWN_ZONES} agent_spawn_zones(grid)={AGENT_SPAWN_ZONES} delivery_zone(grid)={DELIVERY_ZONE}")
print(f"[SETTINGS] obstacles(grid)={sorted(list(OBSTACLES))}")
