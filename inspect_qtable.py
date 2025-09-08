# inspect_qtable.py
import pickle, os, itertools, sys

fn = "qtable.pkl"
print("Archivo:", fn)
if not os.path.exists(fn):
    print("No existe el fichero", fn)
    sys.exit(0)

try:
    d = pickle.load(open(fn, "rb"))
except Exception as e:
    print("Error cargando pickle:", e)
    sys.exit(1)

print("Tipo cargado:", type(d))
try:
    n = len(d)
except Exception:
    n = "N/A"
print("Entradas (len):", n)

# mostrar 5 claves de ejemplo y tamaño de sus vectores
print("\nEjemplos (hasta 5):")
for i, k in enumerate(itertools.islice(d.keys(), 5)):
    v = d[k]
    print(f" {i+1}) key={k}  len(val)={(len(v) if hasattr(v,'__len__') else 'unk')}")
