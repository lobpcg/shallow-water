""" shallow_water_simple_jax.py

2D shallow water model with Coriolis force (f-plane) in JAX.
"""
# To make all JAX variables in float64, enable 64-bit types in JAX
# as they are disabled by default. 
import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import matplotlib.pyplot as plt

# set parameters
n_x = 100
dx = 20e3

n_y = 101
dy = 20e3

gravity = 9.81
depth = 100.0
coriolis_param = 2e-4

dt = 0.5 * min(dx, dy) / jnp.sqrt(gravity * depth)

phase_speed = jnp.sqrt(gravity * depth)
rossby_radius = jnp.sqrt(gravity * depth) / coriolis_param

# plot parameters
plot_range = 0.5
plot_every = 2
max_quivers = 21

# grid setup
x, y = (jnp.arange(n_x) * dx, jnp.arange(n_y) * dy)
Y, X = jnp.meshgrid(y, x, indexing="ij")

# initial conditions
h0 = depth + 1.0 * jnp.exp(
    -((X - x[n_x // 2]) ** 2) / rossby_radius**2
    - (Y - y[n_y - 2]) ** 2 / rossby_radius**2
)
u0 = jnp.zeros_like(h0)
v0 = jnp.zeros_like(h0)

def prepare_plot():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cs = update_plot(0, h0, u0, v0, ax)
    plt.colorbar(cs, label="$\\eta$ (m)")
    return fig, ax

def update_plot(t, h, u, v, ax):
    eta = h - depth

    quiver_stride = (slice(1, -1, n_y // max_quivers), slice(1, -1, n_x // max_quivers))

    ax.clear()
    cs = ax.pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        eta[1:-1, 1:-1],
        vmin=-plot_range,
        vmax=plot_range,
        cmap="RdBu_r",
    )

    if jnp.any((u[quiver_stride] != 0) | (v[quiver_stride] != 0)):
        ax.quiver(
            x[quiver_stride[1]] / 1e3,
            y[quiver_stride[0]] / 1e3,
            u[quiver_stride],
            v[quiver_stride],
            clip_on=False,
        )


    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (km)")
    ax.set_ylabel("$y$ (km)")
    ax.set_title(
        "t=%5.2f days, R=%5.1f km, c=%5.1f m/s "
        % (t / 86400, rossby_radius / 1e3, phase_speed)
    )
    plt.pause(0.1)
    return cs


def iterate_shallow_water():
    # allocate arrays
    u, v, h = jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x))

    # initial conditions
    h = h0
    u = u0
    v = v0

    # boundary values of h must not be used
    h = jnp.pad(h[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=jnp.nan)

    # time step equations
    while True:
        # update u
        v_avg = 0.25 * (v[1:-1, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[:-2, 2:])
        u = u.at[1:-1, 1:-1].set(u[1:-1, 1:-1] + dt * (
            +coriolis_param * v_avg - gravity * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx
        ))
        u = u.at[:, -2].set(0)

        # update v
        u_avg = 0.25 * (u[1:-1, 1:-1] + u[1:-1, :-2] + u[2:, 1:-1] + u[2:, :-2])
        v = v.at[1:-1, 1:-1].set(v[1:-1, 1:-1] + dt * (
            -coriolis_param * u_avg - gravity * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy
        ))
        v = v.at[-2, :].set(0)

        # update h
        h = h.at[1:-1, 1:-1].set(h[1:-1, 1:-1] - dt * depth * (
            (u[1:-1, 1:-1] - u[1:-1, :-2]) / dx + (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dy
        ))

        yield h, u, v


if __name__ == "__main__":
    fig, ax = prepare_plot()

    model = iterate_shallow_water()
    for iteration, (h, u, v) in enumerate(model):
        if iteration % plot_every == 0:
            t = iteration * dt
            update_plot(t, h, u, v, ax)

        # stop if user closes plot window
        if not plt.fignum_exists(fig.number):
            break
