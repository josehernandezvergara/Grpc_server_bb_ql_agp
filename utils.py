# utils.py
# utilidades de rejilla y conversiones
from settings import GRID, BATTERY_BINS, ORIGIN, CELL_SIZE

def clamp(v, a, b):
    return max(a, min(b, v))

def grid_to_world(cell):
    """convierte (gx,gz) -> [x, y, z] world.
    y se fija en 0.5 para match con visualizacion en unity.
    """
    gx, gz = cell
    ox, oz = ORIGIN
    x = (gx - ox) * CELL_SIZE
    z = (gz - oz) * CELL_SIZE
    return [float(x), 0.5, float(z)]

def world_to_grid(pos):
    """convierte [x, y, z] world -> (gx, gz) grid indices.
    redondea y clampa al rango [0, GRID-1].
    """
    wx = float(pos[0])
    wz = float(pos[2])
    ox, oz = ORIGIN
    gx = int(round((wx / CELL_SIZE) + ox))
    gz = int(round((wz / CELL_SIZE) + oz))
    gx = clamp(gx, 0, GRID-1)
    gz = clamp(gz, 0, GRID-1)
    return (gx, gz)

def battery_to_bin(batt):
    idx = int(round((batt/100.0) * (BATTERY_BINS-1)))
    return clamp(idx, 0, BATTERY_BINS-1)
