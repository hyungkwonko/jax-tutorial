import jax.numpy as jnp
from jax import jit, vmap, random
from timeit import timeit

N = [100]
KEY = random.PRNGKey(0)
MAT = random.normal(KEY, (150, 100))
BATCHED_X = random.normal(KEY, (10, 100))

def apply_matrix(v, mat=MAT):
	return jnp.dot(mat, v)

def naively_batched_apply_matrix(batched_x=BATCHED_X):
	return jnp.stack([apply_matrix(v) for v in batched_x])

@jit
def vmap_batched_apply_matrix(batched_x=BATCHED_X):
	return vmap(apply_matrix)(batched_x)  # do not need to care about batch dim


if __name__ == '__main__':
	for n in N:
		print(f"Naively batched... time taken for {n} times: {timeit(naively_batched_apply_matrix, number=n)} sec...")
		print(f"Auto-vectorized with vmap... time taken for {n} times: {timeit(vmap_batched_apply_matrix, number=n)} sec...")
