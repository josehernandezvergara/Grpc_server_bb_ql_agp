# grpc_service.py
# implementacion del servicio grpc que expone getcoords, pickup, drop y assigntask

import time
import random
import warehouse_pb2
import warehouse_pb2_grpc
from utils import grid_to_world, world_to_grid, clamp
from settings import GRID, WAIT_ZONE, ZONE_RADIUS
from qlearning import qlearn

class WarehouseService(warehouse_pb2_grpc.WarehouseServiceServicer):
    def __init__(self, model):
        self.model = model

    def GetCoords(self, request, context):
        # cada llamada avanza 1 step del modelo
        self.model.step()
        objects = []
        
        print(f"[GetCoords] Workers: {len(self.model.workers)}, Boxes: {len(self.model.boxes)}")
        
        # agentes
        for ag in self.model.workers:
            print(f"[GetCoords] Agent{ag.id} at {ag.pos}")
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
                    print(f"[GetCoords] Box{box.id} carried by Agent{box.carried_by} at {bx}, {by}, {bz}")
                else:
                    if getattr(box, "grid_pos", None) is not None:
                        bx, by, bz = grid_to_world(box.grid_pos)
                        print(f"[GetCoords] Box{box.id} at grid {box.grid_pos} -> world {bx}, {by}, {bz}")
                    else:
                        bx, by, bz = box.pos
                        print(f"[GetCoords] Box{box.id} at default pos {bx}, {by}, {bz}")
            else:
                if getattr(box, "grid_pos", None) is not None:
                    bx, by, bz = grid_to_world(box.grid_pos)
                    print(f"[GetCoords] Box{box.id} free at grid {box.grid_pos} -> world {bx}, {by}, {bz}")
                else:
                    bx, by, bz = box.pos
                    print(f"[GetCoords] Box{box.id} free at default pos {bx}, {by}, {bz}")

            obj = warehouse_pb2.ObjectData(
                id=f"Box{box.id}",
                position=warehouse_pb2.Position(x=bx, y=by, z=bz),
                speed=0.5
            )
            objects.append(obj)

        # obstaculos (agregar al mismo array para que unity solo consuma getcoords)
        for obs in getattr(self.model, "obstacles", []):
            try:
                wx, wy, wz = grid_to_world(obs)
                print(f"[GetCoords] Obstacle at grid {obs} -> world {wx}, {wy}, {wz}")
            except Exception:
                if isinstance(obs, (list, tuple)) and len(obs) >= 3:
                    wx = float(obs[0]); wy = float(obs[1]); wz = float(obs[2])
                    print(f"[GetCoords] Obstacle at world {wx}, {wy}, {wz}")
                else:
                    continue
            obj = warehouse_pb2.ObjectData(
                id=f"Obstacle{obs[0]}_{obs[1]}",
                position=warehouse_pb2.Position(x=wx, y=wy, z=wz),
                speed=0.0
            )
            objects.append(obj)

        print(f"[GetCoords] Returning {len(objects)} objects")
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
        
        print(f"[Drop] Agent{aid} entregando Box{bid} en {pos.x}, {pos.y}, {pos.z}")
        
        box = next((b for b in self.model.boxes if b.id == bid), None)
        if box:
            gx, gz = world_to_grid([pos.x, pos.y, pos.z])
            box.grid_pos = (gx, gz)
            box.pos = grid_to_world(box.grid_pos)
            box.carried_by = None
            
        # IMPORTANTE: Completar la tarea ANTES de limpiar el agente
        self.model.blackboard.complete_task_for_box(bid)
        
        ag = self.model.workers_dict.get(aid)
        if ag:
            ag.carrying = False
            ag.carrying_box_id = None
            # CRÍTICO: Limpiar la tarea del agente
            ag.task = None
            
            # ASEGURAR que el agente vuelva a wait_zone
            wx = clamp(WAIT_ZONE[0] + random.randint(-ZONE_RADIUS, ZONE_RADIUS), 0, GRID-1)
            wz = clamp(WAIT_ZONE[1] + random.randint(-ZONE_RADIUS, ZONE_RADIUS), 0, GRID-1)
            ag.roam_target = (wx, wz)
            
            print(f"[Drop] Agent{aid} liberado, nuevo roam_target: {ag.roam_target}")
            
            # Reset de flags de logging
            if hasattr(ag, '_logged_no_task'):
                ag._logged_no_task = False
        
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
        obstacles = []
        for obs in getattr(self.model, "obstacles", set()):
            gx, gz = obs
            world = grid_to_world((gx, gz))
            o = warehouse_pb2.Obstacle()
            try:
                setattr(o, "x", float(world[0]))
            except Exception:
                pass
            try:
                setattr(o, "z", float(world[2]))
            except Exception:
                pass
            try:
                setattr(o, "y", float(world[1]))
            except Exception:
                pass
            obstacles.append(o)
        return warehouse_pb2.ObstaclesList(obstacles=obstacles)
