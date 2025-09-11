# grpc_service.py
# implementacion del servicio grpc que expone getcoords, pickup, drop y assigntask
# comentarios en minuscula y sin acentos

import time
import warehouse_pb2
import warehouse_pb2_grpc
from utils import grid_to_world, world_to_grid
from settings import GRID
from qlearning import qlearn

class WarehouseService(warehouse_pb2_grpc.WarehouseServiceServicer):
    def __init__(self, model):
        self.model = model

    def GetCoords(self, request, context):
        # cada llamada avanza 1 step del modelo
        self.model.step()
        objects = []
        # agentes
        for ag in self.model.workers:
            obj = warehouse_pb2.ObjectData(
                id=f"Agent{ag.id}",
                position=warehouse_pb2.Position(x=ag.pos[0], y=ag.pos[1], z=ag.pos[2]),
                speed=1.0
            )
            objects.append(obj)

        # cajas: si estan siendo llevadas, reportar en la posicion del agente
        for box in self.model.boxes:
            if getattr(box, "carried_by", None) is not None:
                carrier = self.model.workers_dict.get(box.carried_by, None)
                if carrier:
                    bx, by, bz = carrier.pos
                else:
                    if getattr(box, "grid_pos", None) is not None:
                        bx, by, bz = grid_to_world(box.grid_pos)
                    else:
                        bx, by, bz = box.pos
            else:
                if getattr(box, "grid_pos", None) is not None:
                    bx, by, bz = grid_to_world(box.grid_pos)
                else:
                    bx, by, bz = box.pos

            obj = warehouse_pb2.ObjectData(
                id=f"Box{box.id}",
                position=warehouse_pb2.Position(x=bx, y=by, z=bz),
                speed=0.5
            )
            objects.append(obj)

        # obstaculos: añadimos a getcoords en coordenadas world para unity
        for obs in getattr(self.model, "obstacles", []):
            try:
                wx, wy, wz = grid_to_world(obs)
            except Exception:
                # si no es tupla grid, ignorar
                continue
            obj = warehouse_pb2.ObjectData(
                id=f"Obstacle{obs[0]}_{obs[1]}",
                position=warehouse_pb2.Position(x=wx, y=wy, z=wz),
                speed=0.0
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
            self.model.blackboard.assignments[bid] = aid
            box = next((b for b in self.model.boxes if b.id == bid), None)
            if box:
                box.carried_by = aid
                box.grid_pos = None
                box.pos = ag.pos[:]
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
            box.carried_by = None
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
        target = (max(0, min(GRID-1, tx)), max(0, min(GRID-1, tz)))
        self.model.blackboard.assign_task_to_agent(aid, bid, target)
        return warehouse_pb2.Ack(ok=True)

    def GetObstacles(self, request, context):
        # devuelve indices de grid (x= gx, y= gz) segun proto
        obstacles = []
        for obs in getattr(self.model, "obstacles", set()):
            gx, gz = obs
            o = warehouse_pb2.Obstacle()
            o.x = int(gx)
            o.y = int(gz)
            obstacles.append(o)
        return warehouse_pb2.ObstaclesList(obstacles=obstacles)
