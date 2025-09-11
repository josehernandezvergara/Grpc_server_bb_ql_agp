#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador integrado con gRPC
Conecta al servidor gRPC para obtener posiciones en tiempo real
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import time
import os
from datetime import datetime

# Importaciones gRPC con manejo de errores
try:
    import grpc
    import warehouse_pb2
    import warehouse_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  gRPC modules not available: {e}")
    GRPC_AVAILABLE = False

class LiveWarehouseVisualizer:
    def __init__(self, config_file="q_config.json", grpc_host="localhost", grpc_port=50051):
        """Inicializa el visualizador con conexión gRPC"""
        
        # Cargar configuración
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Parámetros del warehouse
        self.grid_size = self.config["grid_size"]
        self.cell_size = self.config["cell_size"]
        self.origin = self.config["origin"]
        self.zone_radius = self.config["zone_radius"]
        
        # Zonas
        self.drop_zone = self.config["drop_zone"]
        self.wait_zone = self.config["wait_zone"]
        self.load_zone = self.config["load_zone"]
        
        # Conexión gRPC
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.channel = None
        self.stub = None
        
        # Configurar matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('Live Warehouse Monitor')
        
        # Historial de posiciones
        self.agent_history = {}
        self.max_history = 20
        
        print(f"✓ Visualizador inicializado")
        print(f"  gRPC: {grpc_host}:{grpc_port}")
        
    def connect_grpc(self):
        """Conecta al servidor gRPC"""
        if not GRPC_AVAILABLE:
            print("❌ Módulos gRPC no disponibles")
            return False
            
        try:
            self.channel = grpc.insecure_channel(f'{self.grpc_host}:{self.grpc_port}')
            self.stub = warehouse_pb2_grpc.WarehouseServiceStub(self.channel)
            
            # Test de conexión
            request = warehouse_pb2.Empty()
            response = self.stub.GetCoords(request)
            
            print(f"✓ Conectado al servidor gRPC")
            print(f"  Workers: {len([obj for obj in response.objects if obj.type == 'worker'])}")
            print(f"  Boxes: {len([obj for obj in response.objects if obj.type == 'box'])}")
            return True
            
        except Exception as e:
            print(f"❌ Error conectando a gRPC: {e}")
            return False
    
    def get_live_data(self):
        """Obtiene datos en tiempo real del servidor"""
        if not GRPC_AVAILABLE:
            return [], []
            
        try:
            request = warehouse_pb2.Empty()
            response = self.stub.GetCoords(request)
            
            agents = []
            boxes = []
            
            for obj in response.objects:
                if obj.type == 'worker':
                    agents.append({
                        'name': obj.name,
                        'x': obj.x,
                        'z': obj.z,
                        'carrying': obj.carrying_object != ""
                    })
                elif obj.type == 'box':
                    boxes.append({
                        'name': obj.name,
                        'x': obj.x,
                        'z': obj.z,
                        'carried_by': obj.carried_by if hasattr(obj, 'carried_by') else ""
                    })
            
            return agents, boxes
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return [], []
    
    def grid_to_world(self, grid_x, grid_z):
        """Convierte coordenadas de grid a world"""
        world_x = (grid_x - self.origin[0]) * self.cell_size
        world_z = (grid_z - self.origin[1]) * self.cell_size
        return world_x, world_z
    
    def draw_zone(self, zone_grid, color, label, alpha=0.3):
        """Dibuja una zona del grid"""
        center_x, center_z = self.grid_to_world(zone_grid[0], zone_grid[1])
        size = (self.zone_radius * 2 + 1) * self.cell_size
        
        rect = patches.Rectangle(
            (center_x - size/2, center_z - size/2), 
            size, size,
            linewidth=2, 
            edgecolor=color, 
            facecolor=color, 
            alpha=alpha
        )
        self.ax.add_patch(rect)
        
        # Etiqueta
        self.ax.text(center_x, center_z + size/2 + 1.5, 
                    f'{label}\\nGrid({zone_grid[0]},{zone_grid[1]})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def update_agent_history(self, agents):
        """Actualiza el historial de posiciones de agentes"""
        for agent in agents:
            name = agent['name']
            pos = (agent['x'], agent['z'])
            
            if name not in self.agent_history:
                self.agent_history[name] = []
            
            # Evitar duplicados
            if not self.agent_history[name] or self.agent_history[name][-1] != pos:
                self.agent_history[name].append(pos)
            
            # Limitar historial
            if len(self.agent_history[name]) > self.max_history:
                self.agent_history[name] = self.agent_history[name][-self.max_history:]
    
    def draw_trajectories(self):
        """Dibuja trayectorias de agentes"""
        colors = ['blue', 'green', 'purple', 'orange']
        
        for i, (agent_name, trajectory) in enumerate(self.agent_history.items()):
            if len(trajectory) > 1:
                xs, zs = zip(*trajectory)
                color = colors[i % len(colors)]
                self.ax.plot(xs, zs, color=color, alpha=0.6, linewidth=2, 
                           linestyle='--', label=f'{agent_name} path')
    
    def update_display(self, agents, boxes):
        """Actualiza la visualización"""
        self.ax.clear()
        
        # Configurar ejes
        self.ax.set_xlim(-55, 55)
        self.ax.set_ylim(-55, 55)
        self.ax.set_xlabel('World X')
        self.ax.set_ylabel('World Z')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Título con timestamp y stats
        current_time = datetime.now().strftime("%H:%M:%S")
        free_agents = len([a for a in agents if not a['carrying']])
        carrying_agents = len([a for a in agents if a['carrying']])
        free_boxes = len([b for b in boxes if not b['carried_by']])
        
        title = f'Live Warehouse Monitor - {current_time}\\n'
        title += f'Agents: {free_agents} free, {carrying_agents} carrying | Boxes: {free_boxes} free'
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Dibujar zonas
        self.draw_zone(self.wait_zone, 'green', 'WAIT')
        self.draw_zone(self.drop_zone, 'red', 'DROP')
        self.draw_zone(self.load_zone, 'blue', 'LOAD')
        
        # Actualizar y dibujar trayectorias
        self.update_agent_history(agents)
        self.draw_trajectories()
        
        # Dibujar agentes
        for agent in agents:
            color = 'orange' if agent['carrying'] else 'blue'
            marker = 's' if agent['carrying'] else 'o'
            size = 100 if agent['carrying'] else 80
            
            self.ax.scatter(agent['x'], agent['z'], c=color, s=size, marker=marker,
                           edgecolors='black', linewidth=1.5, alpha=0.9, zorder=10)
            
            # Etiqueta del agente
            status = '📦' if agent['carrying'] else '🚶'
            self.ax.text(agent['x'] + 0.8, agent['z'] + 0.8, 
                        f"{agent['name']} {status}", 
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor='lightblue', alpha=0.8))
        
        # Dibujar cajas libres
        for box in boxes:
            if not box['carried_by']:  # Solo cajas libres
                self.ax.scatter(box['x'], box['z'], c='red', s=70, marker='^',
                               edgecolors='black', linewidth=1.5, alpha=0.9, zorder=9)
                
                self.ax.text(box['x'] + 0.8, box['z'] - 0.8, 
                            f"📦 {box['name']}", 
                            fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", 
                                    facecolor='lightcoral', alpha=0.8))
        
        # Leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                       markersize=8, label='Agent (libre)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                       markersize=8, label='Agent (cargando)'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                       markersize=8, label='Box (libre)'),
            patches.Patch(color='green', alpha=0.3, label='Wait Zone'),
            patches.Patch(color='red', alpha=0.3, label='Drop Zone'),
            patches.Patch(color='blue', alpha=0.3, label='Load Zone')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def run_live_monitor(self, update_interval=1.0):
        """Ejecuta el monitor en tiempo real"""
        if not self.connect_grpc():
            print("❌ No se pudo conectar al servidor")
            return
        
        print("🚀 Iniciando monitor en tiempo real...")
        print("   Presiona Ctrl+C para salir")
        
        try:
            while True:
                # Obtener datos del servidor
                agents, boxes = self.get_live_data()
                
                if agents or boxes:
                    # Actualizar visualización
                    self.update_display(agents, boxes)
                else:
                    print("⚠️  No hay datos disponibles")
                
                # Esperar antes de la siguiente actualización
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\\n✓ Monitor terminado")
        except Exception as e:
            print(f"\\n❌ Error: {e}")
        finally:
            if self.channel:
                self.channel.close()
            plt.close()

def main():
    """Función principal"""
    print("=== Live Warehouse Visualizer ===")
    
    # Verificar archivos necesarios
    required_files = ["q_config.json"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Error: No se encontró {file}")
            return
    
    if not GRPC_AVAILABLE:
        print("❌ Error: Módulos gRPC no disponibles")
        print("   Instala con: pip install grpcio grpcio-tools")
        return
    
    # Crear y ejecutar visualizador
    visualizer = LiveWarehouseVisualizer()
    visualizer.run_live_monitor(update_interval=0.5)  # Actualizar cada 0.5 segundos

if __name__ == "__main__":
    main()
