# https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
import numpy as np
import jax.numpy as jnp
from jax import random, device_put
from timeit import timeit


N = [1, 10, 100]
SIZE = 3000

def multiply_npy(key=random.PRNGKey(0), size=SIZE):
    x = random.normal(key, (size, size), dtype=np.float32)
    np.dot(x, x.T)

# work like a regular NumPy arrays.
def multiply_jax(key=random.PRNGKey(0), size=SIZE):
    x = random.normal(key, (size, size), dtype=jnp.float32)
    jnp.dot(x, x.T).block_until_ready()  # runs on the GPU  # block_until_ready: to prevent asynchtonous dispatch

def multiply_jax_device_put(key=random.PRNGKey(0), size=SIZE):
    x = random.normal(key, (size, size), dtype=jnp.float32)
    x = device_put(x)
    jnp.dot(x, x.T).block_until_ready()  # runs on the GPU  # block_until_ready: to prevent asynchtonous dispatch

if __name__ == '__main__':
    for n in N:
        print(f"time taken for {n} times: {timeit(multiply_npy, number=n)} sec...")
        print(f"time taken for {n} times: {timeit(multiply_jax, number=n)} sec...")
        print(f"time taken for {n} times: {timeit(multiply_jax_device_put, number=n)} sec...")

