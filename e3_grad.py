import jax.numpy as jnp
from jax import grad


def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

def first_finite_differences(f, x, eps=1e-3):
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in jnp.eye(len(x))])

if __name__ == '__main__':
    x_small = jnp.arange(3.)
    derivative_fn = grad(sum_logistic)
    diff = grad(sum_logistic)(x_small) - first_finite_differences(sum_logistic, x_small)

    print(f"derivative result: {grad(sum_logistic)(x_small)}")
    print(f"first finite differences: {first_finite_differences(sum_logistic, x_small)}")
    print(f"diff: {diff}")
