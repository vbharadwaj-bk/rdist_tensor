import numpy as np
from numpy.random import Generator, Philox


gen = Philox(42)
rg = Generator(gen)
first = gen.random_raw(5)
print(f'First: {first}')


gen = Philox(42)

for i in range(5):
    gen = gen.advance(i)
    rg = Generator(gen)
    sec = gen.random_raw(5)

    #if np.linalg.norm(sec - first) > 1e-7:
    #    print("Not equal!")
    #else:
    #    print("Equal!")

    print(sec)


    #print(f'Second: {rg.random(4)}')

