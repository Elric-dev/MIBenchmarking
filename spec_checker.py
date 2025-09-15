import psutil, os
print("cpus_logical =", psutil.cpu_count())
print("cpus_physical=", psutil.cpu_count(logical=False))
print("mem_gb =", round(psutil.virtual_memory().total/1e9,2))
print("OMP", os.environ.get("OMP_NUM_THREADS"))
print("OPENBLAS", os.environ.get("OPENBLAS_NUM_THREADS"))

