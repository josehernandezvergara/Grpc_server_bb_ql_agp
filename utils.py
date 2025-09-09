# utils.py
# utilidades de rejilla y conversiones
# editar: conversiones entre grid y mundo si tu escena usa otra escala
# nota: usa ORIGIN y CELL_SIZE definidos en settings.py
# comentarios en minuscula y sin acentos
#
# descripcion de funciones:
# - clamp(v,a,b): limita v al rango [a,b]
# - grid_to_world(cell): convierte (gx,gz) a [x,y,z] world; usa ORIGIN y CELL_SIZE
# - world_to_grid(pos): convierte [x,y,z] world a (gx,gz) grid
# - battery_to_bin(batt): discretiza bateria [0..100] en BATTERY_BINS indices


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
