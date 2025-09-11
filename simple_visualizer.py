#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador simplificado sin dependencias gRPC
Lee directamente los logs del servidor o archivos de estado
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import time
import os
import re
from datetime import datetime

class SimpleWarehouseVisualizer:
    def __init__(self, config_file="q_config.json"):
        """Inicializa el visualizador simple"""
        
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
        
        # Configurar matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('Simple Warehouse Monitor')
        
        # Datos actuales
        self.agents = []
        self.boxes = []
        
        print(f"✓ Visualizador simple inicializado")
        print(f"  Grid: {self.grid_size}x{self.grid_size}, Origin: {self.origin}")
        
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
                    f'{label}\nGrid({zone_grid[0]},{zone_grid[1]})\nWorld({center_x:.0f},{center_z:.0f})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def create_sample_data(self):
        """Crea datos de ejemplo basados en tu configuración actual"""
        # Agentes en wait zone
        wait_x, wait_z = self.grid_to_world(self.wait_zone[0], self.wait_zone[1])
        
        agents = [
            {
                'name': 'Agent1',
                'x': wait_x,
                'z': wait_z,
                'carrying': False,
                'state': 'roaming'
            },
            {
                'name': 'Agent2', 
                'x': wait_x + 1,
                'z': wait_z - 1,
                'carrying': False,
                'state': 'roaming'
            },
            {
                'name': 'Agent3',
                'x': wait_x - 1,
                'z': wait_z + 1,
                'carrying': False,
                'state': 'roaming'
            }
        ]
        
        # Cajas en las nuevas posiciones
        boxes = [
            {
                'name': 'Box4',
                'x': self.config["box_spawn_zones_world"][0][0],  # -10
                'z': self.config["box_spawn_zones_world"][0][2],  # 35
                'carried_by': ""
            },
            {
                'name': 'Box5',
                'x': self.config["box_spawn_zones_world"][1][0],  # -15
                'z': self.config["box_spawn_zones_world"][1][2],  # 45
                'carried_by': ""
            }
        ]
        
        return agents, boxes
    
    def update_display(self, agents=None, boxes=None):
        """Actualiza la visualización"""
        if agents is None:
            agents, boxes = self.create_sample_data()
        
        self.ax.clear()
        
        # Configurar ejes
        self.ax.set_xlim(-55, 55)
        self.ax.set_ylim(-55, 55)
        self.ax.set_xlabel('World X')
        self.ax.set_ylabel('World Z')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Título con información
        current_time = datetime.now().strftime("%H:%M:%S")
        free_agents = len([a for a in agents if not a['carrying']])
        carrying_agents = len([a for a in agents if a['carrying']])
        free_boxes = len([b for b in boxes if not b['carried_by']])
        
        title = f'Warehouse Monitor - {current_time}\n'
        title += f'Config: drop_zone{self.drop_zone}, Agents: {free_agents} free, {carrying_agents} carrying | Boxes: {free_boxes} free'
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Dibujar zonas
        self.draw_zone(self.wait_zone, 'green', 'WAIT')
        self.draw_zone(self.drop_zone, 'red', 'DROP')
        self.draw_zone(self.load_zone, 'blue', 'LOAD')
        
        # Spawn zones adicionales
        delivery_zone = self.config["delivery_zone_world"]
        self.ax.scatter(delivery_zone[0], delivery_zone[2], c='purple', s=200, 
                       marker='*', edgecolors='black', linewidth=2, alpha=0.7, label='Delivery Zone')
        
        # Agent spawn zones
        for i, spawn in enumerate(self.config["agent_spawn_zones_world"]):
            self.ax.scatter(spawn[0], spawn[2], c='gold', s=100, 
                           marker='h', edgecolors='black', linewidth=1, alpha=0.6)
            self.ax.text(spawn[0], spawn[2] + 2, f'A{i+1}_spawn', fontsize=8, ha='center')
        
        # Box spawn zones
        for i, spawn in enumerate(self.config["box_spawn_zones_world"]):
            self.ax.scatter(spawn[0], spawn[2], c='orange', s=100, 
                           marker='s', edgecolors='black', linewidth=1, alpha=0.6)
            self.ax.text(spawn[0], spawn[2] - 2, f'Box_spawn', fontsize=8, ha='center')
        
        # Dibujar agentes
        for agent in agents:
            color = 'orange' if agent['carrying'] else 'blue'
            marker = 's' if agent['carrying'] else 'o'
            size = 120 if agent['carrying'] else 100
            
            self.ax.scatter(agent['x'], agent['z'], c=color, s=size, marker=marker,
                           edgecolors='black', linewidth=2, alpha=0.9, zorder=10)
            
            # Etiqueta del agente
            status = '📦' if agent['carrying'] else '🚶'
            state = agent.get('state', 'unknown')
            self.ax.text(agent['x'] + 1.5, agent['z'] + 1.5, 
                        f"{agent['name']} {status}\n{state}", 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='lightblue', alpha=0.8))
        
        # Dibujar cajas libres
        for box in boxes:
            if not box['carried_by']:  # Solo cajas libres
                self.ax.scatter(box['x'], box['z'], c='red', s=80, marker='^',
                               edgecolors='black', linewidth=2, alpha=0.9, zorder=9)
                
                self.ax.text(box['x'] + 1, box['z'] - 1, 
                            f"📦 {box['name']}", 
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                    facecolor='lightcoral', alpha=0.8))
        
        # Información de distancias
        info_text = "DISTANCIAS:\n"
        for agent in agents:
            for box in boxes:
                if not box['carried_by']:
                    dist = ((agent['x'] - box['x'])**2 + (agent['z'] - box['z'])**2)**0.5
                    grid_dist = abs(self.world_to_grid(agent['x'], agent['z'])[0] - self.world_to_grid(box['x'], box['z'])[0]) + \
                               abs(self.world_to_grid(agent['x'], agent['z'])[1] - self.world_to_grid(box['x'], box['z'])[1])
                    info_text += f"{agent['name']}-{box['name']}: {grid_dist} grid cells\n"
        
        info_text += f"\nTask assign radius: {self.config['task_assign_radius']} cells"
        
        self.ax.text(-54, 50, info_text, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
        
        # Leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                       markersize=10, label='Agent (libre)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                       markersize=10, label='Agent (cargando)'),
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
    
    def run_static_view(self):
        """Muestra vista estática del warehouse"""
        print("📊 Mostrando vista estática del warehouse...")
        self.update_display()
        
        print("✓ Vista generada. Cierra la ventana cuando termines.")
        plt.show(block=True)  # Bloquea hasta cerrar ventana
    
    def run_animated_demo(self):
        """Ejecuta demo animado con movimiento simulado"""
        print("🎬 Iniciando demo animado...")
        print("   Presiona Ctrl+C para salir")
        
        try:
            step = 0
            while True:
                # Simular movimiento de agentes
                agents, boxes = self.create_sample_data()
                
                # Añadir algo de movimiento aleatorio
                import random
                for agent in agents:
                    if step % 10 < 5:  # Primeros 5 pasos: mover hacia cajas
                        if boxes:
                            target_box = boxes[0]
                            dx = 1 if target_box['x'] > agent['x'] else -1 if target_box['x'] < agent['x'] else 0
                            dz = 1 if target_box['z'] > agent['z'] else -1 if target_box['z'] < agent['z'] else 0
                            agent['x'] += dx * 0.5
                            agent['z'] += dz * 0.5
                    else:  # Últimos 5 pasos: regresar a wait zone
                        wait_x, wait_z = self.grid_to_world(self.wait_zone[0], self.wait_zone[1])
                        dx = 1 if wait_x > agent['x'] else -1 if wait_x < agent['x'] else 0
                        dz = 1 if wait_z > agent['z'] else -1 if wait_z < agent['z'] else 0
                        agent['x'] += dx * 0.3
                        agent['z'] += dz * 0.3
                    
                    # Simular pickup/delivery
                    if step % 10 == 4:  # En paso 4: pickup
                        agent['carrying'] = True
                        agent['state'] = 'carrying'
                    elif step % 10 == 8:  # En paso 8: delivery
                        agent['carrying'] = False
                        agent['state'] = 'returning'
                    elif step % 10 == 0:  # Reiniciar
                        agent['state'] = 'searching'
                
                # Actualizar visualización
                self.update_display(agents, boxes)
                
                # Esperar y siguiente paso
                time.sleep(1.0)
                step += 1
                
        except KeyboardInterrupt:
            print("\n✓ Demo terminado")
            plt.close()

def main():
    """Función principal"""
    print("=== Simple Warehouse Visualizer ===")
    print("Sin dependencias gRPC - Solo para visualización")
    
    if not os.path.exists("q_config.json"):
        print("❌ Error: No se encontró q_config.json")
        return
    
    visualizer = SimpleWarehouseVisualizer()
    
    print("\nOpciones:")
    print("1. Vista estática")
    print("2. Demo animado")
    
    choice = input("Selecciona opción (1-2): ").strip()
    
    if choice == "1":
        visualizer.run_static_view()
    elif choice == "2":
        visualizer.run_animated_demo()
    else:
        print("Opción inválida, mostrando vista estática...")
        visualizer.run_static_view()

if __name__ == "__main__":
    main()
