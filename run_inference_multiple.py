# run_inference_multiple.py
import subprocess, re, statistics, sys

N = 5  # cuántas corridas
results = []
for i in range(N):
    print(f"Run {i+1}/{N} ...")
    out = subprocess.check_output([sys.executable, "run_inference.py"], text=True)
    # imprimir salida completa para inspección
    print(out)
    m = re.search(r"Entregas en inference:\s*(\d+)", out)
    n = int(m.group(1)) if m else 0
    results.append(n)

print("Resultados:", results)
print("Promedio entregas:", statistics.mean(results) if results else 0)
