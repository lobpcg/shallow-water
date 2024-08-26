""" test_shallow_water_simple_vs_jax.py 

Pytest numpy vs jax implemetations of the
2D shallow water model with Coriolis force (f-plane).
Run
"pytest .\test_shallow_water_simple_vs_jax.py"
or without pytest installed just run the module.

The numpy version runs in float64, while the jax version
defaults to float32. To make all JAX variables in float64,
we change the default to float64 by setting the environment
so that the results must be idetical.
"""

import os

# To make all JAX variables in float64, enable 64-bit types in JAX
# as they are disabled by default.
os.environ["JAX_ENABLE_X64"] = "True"

from shallow_water_simple_jax import iterate_shallow_water as iterate_shallow_water_jax
from shallow_water_simple import iterate_shallow_water
from numpy.testing import assert_array_equal
import jax.numpy as jnp
import numpy as np
import pytest


def test_shallow_water_simple_vs_jax():
    # set parameters
    n_x = 100
    dx = 20e3

    n_y = 101
    dy = 20e3

    gravity = 9.81
    depth = 100.0
    coriolis_param = 2e-4
    rossby_radius = np.sqrt(gravity * depth) / coriolis_param

    # grid setup
    x, y = (np.arange(n_x) * dx, np.arange(n_y) * dy)
    Y, X = np.meshgrid(y, x, indexing="ij")

    # initial conditions
    h0 = depth + 1.0 * np.exp(
        -((X - x[n_x // 2]) ** 2) / rossby_radius**2
        - (Y - y[n_y - 2]) ** 2 / rossby_radius**2
    )
    u0 = np.zeros_like(h0)
    v0 = np.zeros_like(h0)

    # run both models for 1000 time steps
    numer_of_time_steps = 1000
    model = iterate_shallow_water()
    for iteration, (h, u, v) in enumerate(model):
        if iteration == numer_of_time_steps:
            break
    original = (h, u, v)
    model = iterate_shallow_water_jax()
    for iteration, (h, u, v) in enumerate(model):
        if iteration == numer_of_time_steps:
            break
    jax = np.array((h, u, v))
    assert_array_equal(original, jax)


if __name__ == "__main__":
    test_shallow_water_simple_vs_jax()
