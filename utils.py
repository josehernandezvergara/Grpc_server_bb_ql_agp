# utils.py
# utilidades de rejilla y conversiones
# editar: conversiones entre grid y mundo si tu escena usa otra escala

from settings import GRID, BATTERY_BINS

def clamp(v, a, b):
    return max(a, min(b, v))

def grid_to_world(cell):
    # conversion simple: cada celda = 1 unidad, y=0.5 como en unity
    x, y = cell
    return [float(x), 0.5, float(y)]

def world_to_grid(pos):
    x = int(round(pos[0]))
    z = int(round(pos[2]))
    x = clamp(x, 0, GRID-1)
    z = clamp(z, 0, GRID-1)
    return (x, z)

def battery_to_bin(batt):
    idx = int(round((batt/100.0) * (BATTERY_BINS-1)))
    return clamp(idx, 0, BATTERY_BINS-1)
