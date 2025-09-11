#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo rápido del visualizador de warehouse
Genera una imagen estática con todas las zonas
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

def create_warehouse_visualization():
    """Crea una visualización estática del warehouse"""
    
    # Cargar configuración
    with open("q_config.json", 'r') as f:
        config = json.load(f)
    
    # Parámetros del grid
    grid_size = config["grid_size"]
    cell_size = config["cell_size"]
    origin = config["origin"]  # [50, 50]
    
    # Zonas del warehouse
    drop_zone = config["drop_zone"]      # [11, 11]
    wait_zone = config["wait_zone"]      # [0, 0]
    load_zone = config["load_zone"]      # [0, 11]
    zone_radius = config["zone_radius"]  # 2
    
    # Spawn zones
    agent_spawn_zones = config["agent_spawn_zones_world"]
    box_spawn_zones = config["box_spawn_zones_world"]
    delivery_zone = config["delivery_zone_world"]
    
    def grid_to_world(grid_x, grid_z):
        """Convierte coordenadas de grid a world"""
        world_x = (grid_x - origin[0]) * cell_size
        world_z = (grid_z - origin[1]) * cell_size
        return world_x, world_z
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Configurar ejes
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_xlabel('World X (Unity coordinates)', fontsize=12)
    ax.set_ylabel('World Z (Unity coordinates)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Título
    ax.set_title('Warehouse Zones Layout - Configuración Actual', fontsize=16, fontweight='bold', pad=20)
    
    # Función para dibujar zonas del grid
    def draw_zone(zone_grid, color, label, alpha=0.4):
        center_x, center_z = grid_to_world(zone_grid[0], zone_grid[1])
        size = (zone_radius * 2 + 1) * cell_size
        
        rect = patches.Rectangle(
            (center_x - size/2, center_z - size/2), 
            size, size,
            linewidth=3, 
            edgecolor=color, 
            facecolor=color, 
            alpha=alpha
        )
        ax.add_patch(rect)
        
        # Etiqueta con información detallada
        label_text = f'{label}\nGrid: ({zone_grid[0]}, {zone_grid[1]})\nWorld: ({center_x:.0f}, {center_z:.0f})\nRadius: {zone_radius} cells'
        ax.text(center_x, center_z + size/2 + 2, label_text,
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor=color))
    
    # Función para dibujar spawn zones
    def draw_spawn_zone(world_pos, color, label, size=2.0):
        x, z = world_pos[0], world_pos[2]
        
        rect = patches.Rectangle(
            (x - size/2, z - size/2), 
            size, size,
            linewidth=2, 
            edgecolor=color, 
            facecolor=color, 
            alpha=0.3
        )
        ax.add_patch(rect)
        
        ax.text(x, z + size/2 + 1, f'{label}\n({x:.0f}, {z:.0f})',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color))
    
    # Dibujar zonas principales
    draw_zone(wait_zone, 'green', 'WAIT ZONE')
    draw_zone(drop_zone, 'red', 'DROP ZONE') 
    draw_zone(load_zone, 'blue', 'LOAD ZONE')
    
    # Dibujar delivery zone
    draw_spawn_zone(delivery_zone, 'purple', 'DELIVERY ZONE', 3.0)
    
    # Dibujar agent spawn zones
    for i, spawn in enumerate(agent_spawn_zones):
        draw_spawn_zone(spawn, 'gold', f'AGENT {i+1} SPAWN', 1.5)
    
    # Dibujar box spawn zones
    for i, spawn in enumerate(box_spawn_zones):
        draw_spawn_zone(spawn, 'orange', f'BOX {i+4} SPAWN', 1.5)
    
    # Simular posiciones actuales basadas en tus logs
    # Agentes en wait zone
    agents = [
        {'name': 'Agent1', 'x': -50.0, 'z': -50.0, 'carrying': False},
        {'name': 'Agent2', 'x': -49.0, 'z': -49.0, 'carrying': False}, 
        {'name': 'Agent3', 'x': -50.0, 'z': -50.0, 'carrying': False}
    ]
    
    # Cajas en drop zone
    boxes = [
        {'name': 'Box4', 'x': -39.0, 'z': -39.0},
        {'name': 'Box5', 'x': -39.0, 'z': -39.0}
    ]
    
    # Dibujar agentes
    for agent in agents:
        color = 'orange' if agent['carrying'] else 'darkblue'
        marker = 's' if agent['carrying'] else 'o'
        
        ax.scatter(agent['x'], agent['z'], c=color, s=120, marker=marker, 
                   edgecolors='black', linewidth=2, alpha=0.9, zorder=10)
        
        ax.text(agent['x'] + 1, agent['z'] + 1, agent['name'], 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # Dibujar cajas
    for box in boxes:
        ax.scatter(box['x'], box['z'], c='darkred', s=100, marker='^', 
                   edgecolors='black', linewidth=2, alpha=0.9, zorder=9)
        
        ax.text(box['x'] + 1, box['z'] - 1, box['name'], 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
    
    # Leyenda detallada
    legend_elements = [
        patches.Patch(color='green', alpha=0.4, label='Wait Zone (Zona de espera)'),
        patches.Patch(color='red', alpha=0.4, label='Drop Zone (Zona de entrega)'),
        patches.Patch(color='blue', alpha=0.4, label='Load Zone (Zona de carga)'),
        patches.Patch(color='purple', alpha=0.3, label='Delivery Zone'),
        patches.Patch(color='gold', alpha=0.3, label='Agent Spawn'),
        patches.Patch(color='orange', alpha=0.3, label='Box Spawn'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                   markersize=10, label='Agent (libre)', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                   markersize=10, label='Agent (cargando)', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='darkred', 
                   markersize=10, label='Box (libre)', markeredgecolor='black')
    ]
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    # Información adicional
    info_text = f"""CONFIGURACIÓN ACTUAL:
Grid Size: {grid_size}x{grid_size}
Cell Size: {cell_size}
Origin: {origin}
Zone Radius: {zone_radius}

COORDENADAS:
Wait Zone: Grid{wait_zone} → World{grid_to_world(wait_zone[0], wait_zone[1])}
Drop Zone: Grid{drop_zone} → World{grid_to_world(drop_zone[0], drop_zone[1])}
Load Zone: Grid{load_zone} → World{grid_to_world(load_zone[0], load_zone[1])}"""
    
    ax.text(-54, 50, info_text, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    # Guardar imagen
    plt.savefig('warehouse_zones_layout.png', dpi=300, bbox_inches='tight')
    plt.savefig('warehouse_zones_layout.pdf', bbox_inches='tight')  # También en PDF
    
    print("✓ Visualización creada:")
    print("  📄 warehouse_zones_layout.png")
    print("  📄 warehouse_zones_layout.pdf")
    
    # Mostrar
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    print("=== Warehouse Zones Visualizer ===")
    print("Generando visualización...")
    
    if not os.path.exists("q_config.json"):
        print("❌ Error: No se encontró q_config.json")
    else:
        create_warehouse_visualization()
        print("✓ Completado!")
