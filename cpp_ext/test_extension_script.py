import numpy as np
import cppimport.import_hook
import redistribute_tensor as rd

if __name__=='__main__':
    size= 5000
    print("Starting rng generation!")
    x = np.random.randint(0, 2, size=500000000)
    x = np.array(x, dtype=np.ulonglong)
    print("Starting redistribution!")
    rd.redistribute_nonzeros([5, 6, 7], [x], 5)

