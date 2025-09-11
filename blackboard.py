# blackboard.py
# lista de tareas y asignaciones entre cajas y agentes
# comentarios en minuscula y sin acentos

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
        if box_id in self.assignments:
            del self.assignments[box_id]
        return False

    def assign_task_to_agent(self, agent_id, box_id, target_grid):
        for task in self.tasks:
            if task["obj_id"] == box_id:
                task["target"] = target_grid
                task["done"] = False
                self.assignments[box_id] = agent_id
                return True
        box = next((b for b in self.model.boxes if b.id == box_id), None)
        start = box.grid_pos if box is not None else (0,0)
        self.add_task(box_id, start, target_grid)
        self.assignments[box_id] = agent_id
        return True
