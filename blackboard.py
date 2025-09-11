# blackboard.py
# filepath: c:\Users\yeric\grpc\blackboard.py
# lista de tareas y asignaciones entre cajas y agentes
# comentarios en minuscula y sin acentos
#
# descripcion:
# - BlackBoard.model: referencia al modelo que la contiene
# - BlackBoard.tasks: lista de diccionarios {obj_id, start, target, done}
# - BlackBoard.assignments: mapping box_id -> agent_id
#
# metodos:
# - add_task(obj_id, start_grid, target_grid, min_dist=0.5): anade una tarea a la lista
# - get_task(agent_id): asigna y retorna la primera tarea no realizada y no asignada
#   pero solo si el agente esta lo suficientemente cerca (task_assign_radius)
# - complete_task_for_box(box_id): marca la tarea como done y libera la asignacion
# - assign_task_to_agent(agent_id, box_id, target_grid): fuerza asignacion y crea tarea si es necesario

from typing import Optional
import math

# import settings para leer TASK_ASSIGN_RADIUS si esta definido
try:
    import settings
    TASK_ASSIGN_RADIUS = getattr(settings, 'TASK_ASSIGN_RADIUS', None)
    # por compatibilidad con nombres alternativos en configs
    if TASK_ASSIGN_RADIUS is None:
        TASK_ASSIGN_RADIUS = getattr(settings, 'task_assign_radius', None)
except Exception:
    TASK_ASSIGN_RADIUS = None

# si no se define, usar un radio muy grande (comportamiento compatibilidad)
if TASK_ASSIGN_RADIUS is None:
    TASK_ASSIGN_RADIUS = 9999


class BlackBoard:
    def __init__(self, model):
        self.model = model
        self.tasks = []        # lista de dicts {obj_id, start, target, done}
        self.assignments = {}  # map box_id -> agent_id

    def add_task(self, obj_id, start_grid, target_grid, min_dist=0.5):
        """anade una tarea a la lista"""
        self.tasks.append({
            "obj_id": obj_id,
            "start": start_grid,
            "target": target_grid,
            "done": False
        })

    def get_task(self, agent_id: int) -> Optional[dict]:
        """
        asigna y retorna la primera tarea no realizada y no asignada.
        ahora se verifica la distancia entre el agente y la caja (manhattan).
        solo se asigna si la distancia <= TASK_ASSIGN_RADIUS.

        retorna None si no hay tareas apropiadas.
        """
        # intentar obtener el agente para chequear su posicion
        agent = None
        try:
            agent = self.model.workers_dict.get(agent_id, None)
        except Exception:
            agent = None

        for task in self.tasks:
            if task.get("done"):
                continue
            obj_id = task.get("obj_id")
            if obj_id in self.assignments:
                continue

            # si no tenemos agente o no hay posicion de caja, asignar por compatibilidad
            if agent is None:
                # asignar sin chequear distancia si no podemos consultarla
                self.assignments[obj_id] = agent_id
                print(f"[BlackBoard] Asignando tarea Box{obj_id} a Agent{agent_id} (sin verificar distancia)")
                return task

            # buscar la caja en el modelo para conocer su posicion
            box = next((b for b in self.model.boxes if b.id == obj_id), None)
            if box is None:
                # si la caja no existe, marcar tarea como done para evitar reintentarlo
                task["done"] = True
                continue

            box_pos = getattr(box, "grid_pos", None)
            agent_pos = getattr(agent, "grid_pos", None)

            # si la caja no tiene posicion (p. ej. ya esta siendo llevada), no asignar
            if box_pos is None:
                continue

            # si por alguna razon no conocemos la posicion del agente, asignar por compatibilidad
            if agent_pos is None:
                self.assignments[obj_id] = agent_id
                print(f"[BlackBoard] Asignando tarea Box{obj_id} a Agent{agent_id} (sin posición del agente)")
                return task

            # calcular distancia manhattan en grid
            dist = abs(int(agent_pos[0]) - int(box_pos[0])) + abs(int(agent_pos[1]) - int(box_pos[1]))

            # comprobar radio configurable
            if dist <= TASK_ASSIGN_RADIUS:
                self.assignments[obj_id] = agent_id
                print(f"[BlackBoard] Asignando tarea Box{obj_id} a Agent{agent_id} (distancia: {dist})")
                return task
            else:
                # no asignamos porque el agente esta muy lejos; seguir buscando otras tareas
                print(f"[BlackBoard] Agent{agent_id} muy lejos de Box{obj_id} (distancia: {dist} > {TASK_ASSIGN_RADIUS})")
                continue

        print(f"[BlackBoard] No hay tareas disponibles para Agent{agent_id}")
        return None

    def complete_task_for_box(self, box_id):
        """marca la tarea como done y libera la asignacion"""
        for task in self.tasks:
            if task["obj_id"] == box_id:
                task["done"] = True
                if box_id in self.assignments:
                    del self.assignments[box_id]
                return True
        # si no hay tarea pero existe la asignacion, limpiar igualmente
        if box_id in self.assignments:
            del self.assignments[box_id]
        return False

    def assign_task_to_agent(self, agent_id, box_id, target_grid):
        """
        fuerza asignacion y crea tarea si es necesario
        """
        for task in self.tasks:
            if task["obj_id"] == box_id:
                task["target"] = target_grid
                task["done"] = False
                self.assignments[box_id] = agent_id
                return True
        # si no existia la tarea, crearla usando la posicion de la caja si esta disponible
        box = next((b for b in self.model.boxes if b.id == box_id), None)
        start = getattr(box, "grid_pos", (0, 0))
        self.add_task(box_id, start, target_grid)
        self.assignments[box_id] = agent_id
        return True
