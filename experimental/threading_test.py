import numpy as np
import time

x = np.ones((10000, 10000))
start = time.time()
y = x @ x
delta = time.time() - start
print(f"Elapsed: {delta}s")