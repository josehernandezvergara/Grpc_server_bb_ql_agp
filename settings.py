# settings.py
# carga de configuracion y constantes compartidas
# editar: valores por defecto, rutas de archivo, zonas, acciones, etc
# nota: todas las variables expuestas aqui son usadas por los demas modulos

import os
import json

CONFIG_PATH = "q_config.json"

default_config = {
    # q-learning hiperparametros basicos (puedes editar aqui o en q_config.json)
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.3,
    "epsilon_decay": 0.9999,
    "min_epsilon": 0.05,

    # ambiente y guardado
    "grid_size": 12,
    "battery_bins": 11,
    "save_every_steps": 1000,
    "qfile": "qtable.pkl",
    "inference_file": "qtable_0_inference.pkl",
    "snapshot_inference_on_save": True,
    "reset_on_train": False,

    # hardware/infra
    "charger_positions": [[0,0]],

    # acciones disponibles (orden fijo)
    "actions": ["N","S","E","W","INTERACT","WAIT"],

    # zonas (coordenadas de grid)
    "drop_zone": [11, 11],
    "wait_zone": [0, 0],
    "load_zone": [0, 11],
    "zone_radius": 1,

    # reward shaping / costes (puedes ajustar aqui)
    "move_cost_base": 0.3,
    "carry_multiplier": 1.01,
    "pickup_reward": 100.0,
    "drop_reward": 400.0,
    "block_penalty": -5.0,
    "wait_penalty": -0.2,
    "interact_nop_penalty": -1.0,
    "battery_zero_penalty": -50.0,
    "distance_reward": 1.0,
    "distance_penalty": -0.5
}

# crear q_config.json si no existe (se escribe valores por defecto)
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
ALPHA = config.get("alpha", default_config["alpha"])
GAMMA = config.get("gamma", default_config["gamma"])
EPSILON = config.get("epsilon", default_config["epsilon"])
EPS_DECAY = config.get("epsilon_decay", default_config["epsilon_decay"])
EPS_MIN = config.get("min_epsilon", default_config["min_epsilon"])
GRID = config.get("grid_size", default_config["grid_size"])
BATTERY_BINS = config.get("battery_bins", default_config["battery_bins"])
SAVE_EVERY = config.get("save_every_steps", default_config["save_every_steps"])
QFILE = config.get("qfile", default_config["qfile"])
INF_FILE = config.get("inference_file", default_config["inference_file"])
SNAPSHOT_INF_ON_SAVE = config.get("snapshot_inference_on_save", default_config["snapshot_inference_on_save"])
RESET_ON_TRAIN = config.get("reset_on_train", default_config["reset_on_train"])
CHARGERS = [tuple(p) for p in config.get("charger_positions", default_config["charger_positions"])]
ACTIONS = config.get("actions", default_config["actions"])
NUM_ACTIONS = len(ACTIONS)

# zonas
DROP_ZONE = tuple(config.get("drop_zone", default_config["drop_zone"]))
WAIT_ZONE = tuple(config.get("wait_zone", default_config["wait_zone"]))
LOAD_ZONE = tuple(config.get("load_zone", default_config["load_zone"]))
ZONE_RADIUS = int(config.get("zone_radius", default_config["zone_radius"]))

# reward shaping (leido desde config para que sea facil ajustar)
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

# modo: se puede setear mediante la variable de entorno MODE
# valores: "train" (por defecto) o "inference"
MODE = os.getenv("MODE", "").strip().lower()
if MODE == "":
    force_inf = os.getenv("FORCE_INFERENCE", "").strip().lower()
    if force_inf in ("1","true","yes"):
        MODE = "inference"
    else:
        MODE = "train"
