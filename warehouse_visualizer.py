#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de zonas del warehouse en Python
Genera gráficos en tiempo real mostrando:
- Zonas del grid (wait, drop, load)
- Posiciones de agentes y cajas
- Spawn zones y delivery zone
- Trayectorias de movimiento

Uso:
    python warehouse_visualizer.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import time
from datetime import datetime
import os

class WarehouseVisualizer:
    def __init__(self, config_file="q_config.json"):
        """Inicializa el visualizador con la configuración del warehouse"""
        
        # Cargar configuración
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Parámetros del grid
        self.grid_size = self.config["grid_size"]
        self.cell_size = self.config["cell_size"]
        self.origin = self.config["origin"]  # [50, 50]
        
        # Zonas del warehouse
        self.drop_zone = self.config["drop_zone"]      # [11, 11]
        self.wait_zone = self.config["wait_zone"]      # [0, 0]
        self.load_zone = self.config["load_zone"]      # [0, 11]
        self.zone_radius = self.config["zone_radius"]  # 2
        
        # Spawn zones (coordenadas world)
        self.agent_spawn_zones = self.config["agent_spawn_zones_world"]
        self.box_spawn_zones = self.config["box_spawn_zones_world"]
        self.delivery_zone = self.config["delivery_zone_world"]
        
        # Configurar matplotlib
        plt.ion()  # Modo interactivo
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('Warehouse Zones Visualizer')
        
        # Datos actuales (simulados, normalmente vendrían del servidor gRPC)
        self.agents = []
        self.boxes = []
        self.trajectories = {}  # Para guardar trayectorias
        
        print(f"✓ Visualizador inicializado")
        print(f"  Grid: {self.grid_size}x{self.grid_size}, Origin: {self.origin}")
        print(f"  Zonas - Wait: {self.wait_zone}, Drop: {self.drop_zone}, Load: {self.load_zone}")
    
    def grid_to_world(self, grid_x, grid_z):
        """Convierte coordenadas de grid a world"""
        world_x = (grid_x - self.origin[0]) * self.cell_size
        world_z = (grid_z - self.origin[1]) * self.cell_size
        return world_x, world_z
    
    def world_to_grid(self, world_x, world_z):
        """Convierte coordenadas world a grid"""
        grid_x = int(world_x / self.cell_size + self.origin[0])
        grid_z = int(world_z / self.cell_size + self.origin[1])
        return grid_x, grid_z
    
    def draw_zone(self, zone_grid, color, label, alpha=0.3):
        """Dibuja una zona del grid"""
        # Convertir a coordenadas world
        center_x, center_z = self.grid_to_world(zone_grid[0], zone_grid[1])
        
        # Tamaño de la zona (radio en celdas)
        size = (self.zone_radius * 2 + 1) * self.cell_size
        
        # Crear rectángulo centrado
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
        self.ax.text(center_x, center_z + size/2 + 1, 
                    f'{label}\nGrid({zone_grid[0]},{zone_grid[1]})\nWorld({center_x:.0f},{center_z:.0f})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def draw_spawn_zone(self, world_pos, color, label, size=1.5):
        """Dibuja una zona de spawn (coordenadas world)"""
        x, z = world_pos[0], world_pos[2]  # world_pos es [x, y, z]
        
        rect = patches.Rectangle(
            (x - size/2, z - size/2), 
            size, size,
            linewidth=1.5, 
            edgecolor=color, 
            facecolor=color, 
            alpha=0.4
        )
        
        self.ax.add_patch(rect)
        
        # Etiqueta
        self.ax.text(x, z + size/2 + 0.5, 
                    f'{label}\n({x:.0f},{z:.0f})',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    def draw_agent(self, agent_data):
        """Dibuja un agente"""
        name = agent_data['name']
        x, z = agent_data['x'], agent_data['z']
        
        # Color según estado
        if agent_data.get('carrying_box'):
            color = 'orange'
            marker = 's'  # cuadrado
            size = 80
        else:
            color = 'blue'
            marker = 'o'  # círculo
            size = 60
        
        self.ax.scatter(x, z, c=color, s=size, marker=marker, 
                       edgecolors='black', linewidth=1, alpha=0.8, zorder=10)
        
        # Etiqueta del agente
        self.ax.text(x + 0.5, z + 0.5, name, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8))
        
        # Guardar para trayectoria
        if name not in self.trajectories:
            self.trajectories[name] = []
        self.trajectories[name].append((x, z))
        
        # Limitar trayectoria a últimos 10 puntos
        if len(self.trajectories[name]) > 10:
            self.trajectories[name] = self.trajectories[name][-10:]
    
    def draw_box(self, box_data):
        """Dibuja una caja"""
        name = box_data['name']
        x, z = box_data['x'], box_data['z']
        
        # Color según estado
        if box_data.get('carried_by'):
            return  # No dibujar si está siendo cargada
        
        color = 'red'
        self.ax.scatter(x, z, c=color, s=60, marker='^', 
                       edgecolors='black', linewidth=1, alpha=0.8, zorder=9)
        
        # Etiqueta
        self.ax.text(x + 0.5, z - 0.5, name, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.8))
    
    def draw_trajectories(self):
        """Dibuja trayectorias de agentes"""
        colors = ['blue', 'green', 'purple']
        for i, (agent_name, trajectory) in enumerate(self.trajectories.items()):
            if len(trajectory) > 1:
                xs, zs = zip(*trajectory)
                color = colors[i % len(colors)]
                self.ax.plot(xs, zs, color=color, alpha=0.6, linewidth=1, linestyle='--')
    
    def update_display(self, agents_data=None, boxes_data=None):
        """Actualiza la visualización"""
        self.ax.clear()
        
        # Configurar ejes
        self.ax.set_xlim(-55, 55)
        self.ax.set_ylim(-55, 55)
        self.ax.set_xlabel('World X')
        self.ax.set_ylabel('World Z')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Título con timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        self.ax.set_title(f'Warehouse Zones Visualization - {current_time}', fontsize=14, fontweight='bold')
        
        # Dibujar zonas principales
        self.draw_zone(self.wait_zone, 'green', 'WAIT ZONE')
        self.draw_zone(self.drop_zone, 'red', 'DROP ZONE')
        self.draw_zone(self.load_zone, 'blue', 'LOAD ZONE')
        
        # Dibujar spawn zones
        self.draw_spawn_zone(self.delivery_zone, 'purple', 'DELIVERY', 2.0)
        
        for i, spawn in enumerate(self.agent_spawn_zones):
            self.draw_spawn_zone(spawn, 'gold', f'AGENT{i+1}', 1.0)
        
        for i, spawn in enumerate(self.box_spawn_zones):
            self.draw_spawn_zone(spawn, 'orange', f'BOX{i+4}', 1.0)
        
        # Dibujar trayectorias
        self.draw_trajectories()
        
        # Dibujar agentes y cajas
        if agents_data:
            for agent in agents_data:
                self.draw_agent(agent)
        
        if boxes_data:
            for box in boxes_data:
                self.draw_box(box)
        
        # Leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Agent (Free)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=8, label='Agent (Carrying)'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Box (Free)'),
            patches.Patch(color='green', alpha=0.3, label='Wait Zone'),
            patches.Patch(color='red', alpha=0.3, label='Drop Zone'),
            patches.Patch(color='blue', alpha=0.3, label='Load Zone')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def simulate_data_from_logs(self):
        """Simula datos basándose en los logs que proporcionaste"""
        # Datos simulados basados en tu output
        agents_data = [
            {
                'name': 'Agent1',
                'x': -50.0,
                'z': -50.0,
                'carrying_box': False
            },
            {
                'name': 'Agent2', 
                'x': -49.0,
                'z': -49.0,
                'carrying_box': False
            },
            {
                'name': 'Agent3',
                'x': -50.0,
                'z': -50.0,
                'carrying_box': False
            }
        ]
        
        boxes_data = [
            {
                'name': 'Box4',
                'x': -39.0,
                'z': -39.0,
                'carried_by': None
            },
            {
                'name': 'Box5',
                'x': -39.0,
                'z': -39.0,
                'carried_by': None
            }
        ]
        
        return agents_data, boxes_data
    
    def run_live_visualization(self):
        """Ejecuta visualización en tiempo real"""
        print("🚀 Iniciando visualización en tiempo real...")
        print("   Presiona Ctrl+C para salir")
        
        try:
            while True:
                # Simular datos (en tu caso real, estos vendrían del servidor gRPC)
                agents_data, boxes_data = self.simulate_data_from_logs()
                
                # Actualizar visualización
                self.update_display(agents_data, boxes_data)
                
                # Esperar un poco
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n✓ Visualización terminada")
            plt.close()
    
    def save_static_visualization(self, filename="warehouse_zones.png"):
        """Guarda una imagen estática del warehouse"""
        agents_data, boxes_data = self.simulate_data_from_logs()
        self.update_display(agents_data, boxes_data)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Imagen guardada en: {filename}")

def main():
    """Función principal"""
    print("=== Warehouse Zones Visualizer ===")
    
    # Verificar que existe el archivo de configuración
    if not os.path.exists("q_config.json"):
        print("❌ Error: No se encontró q_config.json")
        return
    
    # Crear visualizador
    visualizer = WarehouseVisualizer()
    
    # Opciones
    print("\nOpciones:")
    print("1. Visualización en tiempo real")
    print("2. Guardar imagen estática")
    print("3. Solo mostrar zonas")
    
    choice = input("\nSelecciona opción (1-3): ").strip()
    
    if choice == "1":
        visualizer.run_live_visualization()
    elif choice == "2":
        visualizer.save_static_visualization()
        input("Presiona Enter para cerrar...")
    elif choice == "3":
        visualizer.update_display()
        input("Presiona Enter para cerrar...")
    else:
        print("Opción inválida")

if __name__ == "__main__":
    main()
