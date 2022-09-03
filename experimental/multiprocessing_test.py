from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    print("Starting multiprocessing test...")
    with Pool(128) as p:
        print(p.map(f, [1, 2, 3]))