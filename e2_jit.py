import numpy as np
import jax.numpy as jnp
from jax import jit, random
from timeit import timeit


N = [1, 10, 100]
SIZE = 1000000

def selu_npy(size=SIZE, alpha=1.67, lmbda=1.05, key = random.PRNGKey(0)):
    x = random.normal(key, (size,))
    out = lmbda * np.where(x > 0, x, alpha * np.exp(x) - alpha)
    return out

def selu_jax(size=SIZE, alpha=1.67, lmbda=1.05, key = random.PRNGKey(0)):
    x = random.normal(key, (size,))
    out = lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
    return out.block_until_ready()

def selu_jit(func=selu_jax):
    return jit(func)

if __name__ == '__main__':
    for n in N:
        print(f"time taken for {n} times: {timeit(selu_npy, number=n)} sec...")
        print(f"time taken for {n} times: {timeit(selu_jax, number=n)} sec...")
        print(f"time taken for {n} times: {timeit(selu_jit, number=n)} sec...")


