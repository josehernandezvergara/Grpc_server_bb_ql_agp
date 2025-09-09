# qlearning.py
# gestor de q-learning: carga, guarda, elegir accion y actualizar
# editar: hiperparametros en q_config.json via settings.py
# el objeto qlearn se crea al importar el modulo y refleja el modo (train/inference)
#
# descripcion de la clase qlearning y sus metodos:
# - QLearning.Q: diccionario clave->vector de q-values (key = estado, vector len = num acciones)
# - QLearning.epsilon: probabilidad de exploracion (epsilon-greedy)
# - QLearning.alpha, gamma: hiperparametros de aprendizaje
# - QLearning.file, inference_file: rutas para guardar/cargar la tabla
# - QLearning.inference_mode: flag que indica que no se actualiza la Q
#
# metodos principales:
# - _load_file(path): carga pickle serializado y convierte a numpy arrays
# - _ensure(key): asegura que exista un vector q para la key y lo inicializa a zeros
# - state_key(...): forma la tupla de estado usada como clave en la tabla
# - choose_action(state_key): epsilon-greedy, devuelve indice de accion (int)
# - update(s_key,a,r,s2_key): actualizacion q-learning clasica
# - decay_epsilon(): reduce epsilon segun EPS_DECAY hasta EPS_MIN
# - save(write_inference_snapshot=False): guarda tabla de forma atomica y opcional snapshot de inferencia
# - load_inference(): intenta cargar el snapshot de inferencia y fija epsilon a 0


import os
import pickle
import tempfile
import random
import numpy as np

from settings import (QFILE, INF_FILE, SNAPSHOT_INF_ON_SAVE, RESET_ON_TRAIN,
                      EPSILON, ALPHA, GAMMA, EPS_DECAY, EPS_MIN, NUM_ACTIONS, MODE)

class QLearning:
    def __init__(self, file_path=QFILE, inference_path=INF_FILE):
        self.Q = {}
        self.epsilon = EPSILON
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.file = file_path
        self.inference_file = inference_path
        self.inference_mode = False

        if MODE == "inference":
            # intentar cargar snapshot de inferencia primero
            if os.path.exists(self.inference_file):
                self._load_file(self.inference_file)
                self.epsilon = 0.0
                self.decay_epsilon = lambda: None
                self.inference_mode = True
                print(f"[Q] cargada tabla de inferencia: {self.inference_file} entradas={len(self.Q)}")
            else:
                if os.path.exists(self.file):
                    self._load_file(self.file)
                    self.epsilon = 0.0
                    self.decay_epsilon = lambda: None
                    self.inference_mode = True
                    if SNAPSHOT_INF_ON_SAVE:
                        try:
                            self.save(write_inference_snapshot=True)
                        except Exception:
                            pass
                    print("[Q] inference solicitado: se cargo qfile como fallback")
                else:
                    self.Q = {}
                    self.epsilon = 0.0
                    self.decay_epsilon = lambda: None
                    self.inference_mode = True
                    print("[Q] modo inference y no se encontro archivo: tabla vacia iniciada (epsilon=0)")
        else:
            # train
            if RESET_ON_TRAIN:
                self.Q = {}
                print("[Q] modo train: reset_on_train activo -> tabla vacia iniciada")
            else:
                if os.path.exists(self.file):
                    self._load_file(self.file)
                else:
                    print(f"[Q] modo train: no se encontro {self.file}, iniciar con tabla vacia")

    def _load_file(self, path):
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                self.Q = {tuple(k): np.array(v, dtype=np.float32) for k, v in loaded.items()}
                print(f"[Q] cargada {path} entradas={len(self.Q)}")
            else:
                print(f"[Q] formato inesperado en {path}, ignorando")
        except Exception as e:
            print(f"[Q] error cargando {path}: {e}")

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
        if self.inference_mode:
            return
        q_s = self._ensure(s_key)
        q_s2 = self._ensure(s2_key)
        qsa = q_s[a]
        qmax_next = np.max(q_s2)
        q_s[a] = qsa + self.alpha * (r + self.gamma * qmax_next - qsa)

    def decay_epsilon(self):
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    def save(self, write_inference_snapshot=False):
        try:
            serial = {k: v.tolist() for k, v in self.Q.items()}
            dirpath = os.path.dirname(os.path.abspath(self.file)) or "."
            with tempfile.NamedTemporaryFile("wb", delete=False, dir=dirpath) as tf:
                pickle.dump(serial, tf)
                tmpname = tf.name
            os.replace(tmpname, self.file)
            print(f"[Q] q-table guardada en {self.file} entries={len(self.Q)}")
            if write_inference_snapshot or SNAPSHOT_INF_ON_SAVE:
                dirinf = os.path.dirname(os.path.abspath(self.inference_file)) or "."
                with tempfile.NamedTemporaryFile("wb", delete=False, dir=dirinf) as tif:
                    pickle.dump(serial, tif)
                    tmpinf = tif.name
                os.replace(tmpinf, self.inference_file)
                print(f"[Q] snapshot para inferencia guardado en {self.inference_file}")
        except Exception as e:
            print(f"[Q] error guardando q-table: {e}")

    def load_inference(self):
        if os.path.exists(self.inference_file):
            self._load_file(self.inference_file)
            self.epsilon = 0.0
            self.decay_epsilon = lambda: None
            self.inference_mode = True
            print("[Q] inferencia: epsilon fijado a 0.0 desde inference_file")
            return True
        return False

# instancia global
qlearn = QLearning(file_path=QFILE, inference_path=INF_FILE)
