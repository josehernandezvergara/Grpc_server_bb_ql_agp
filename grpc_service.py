# grpc_service.py
# implementacion del servicio grpc que expone getcoords, pickup, drop y assigntask
# no editar la interfaz grpc aqui salvo que se regenere warehouse_pb2.py

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
            self.model.blackboard.assignments[bid] = aid
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
        target = (max(0, min(GRID-1, tx)), max(0, min(GRID-1, tz)))
        self.model.blackboard.assign_task_to_agent(aid, bid, target)
        return warehouse_pb2.Ack(ok=True)
    
    def getObstacles(self, request, context):
        obstacles = [
            warehouse_pb2.Obstacle(x=obs[0], y=obs[1])
            for obs in self.model.obstacles
        ]
        return warehouse_pb2.ObstaclesList(obstacles=obstacles)
