""" model2d_nonlinear.py

2D shallow water model with:

- varying Coriolis force
- nonlinear terms
- lateral friction
- periodic boundary conditions

"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# grid setup
n_x = 200
dx = 5e3
l_x = n_x * dx

n_y = 104
dy = 5e3
l_y = n_y * dy

x, y = (
    jnp.arange(n_x) * dx,
    jnp.arange(n_y) * dy
)
Y, X = jnp.meshgrid(y, x, indexing='ij')

# physical parameters
gravity = 9.81
depth = 100.
coriolis_f = 2e-4
coriolis_beta = 2e-11
coriolis_param = coriolis_f + Y * coriolis_beta
lateral_viscosity = 1e-3 * coriolis_f * dx ** 2

# other parameters
periodic_boundary_x = False
linear_momentum_equation = False

adams_bashforth_a = 1.5 + 0.1
adams_bashforth_b = -(0.5 + 0.1)

dt = 0.125 * min(dx, dy) / jnp.sqrt(gravity * depth)

phase_speed = jnp.sqrt(gravity * depth)
rossby_radius = jnp.sqrt(gravity * depth) / coriolis_param.mean()

# plot parameters
plot_range = 10
plot_every = 10
max_quivers = 41

# initial conditions
u0 = 10 * jnp.exp(-(Y - y[n_y // 2])**2 / (0.02 * l_x)**2)
v0 = jnp.zeros_like(u0)
# approximate balance h_y = -(f/g)u
h_geostrophy = jnp.cumsum(-dy * u0 * coriolis_param / gravity, axis=0)
h0 = (
    depth
    + h_geostrophy
    # make sure h0 is centered around depth
    - h_geostrophy.mean()
    # small perturbation
    + 0.2 * jnp.sin(X / l_x * 10 * jnp.pi) * jnp.cos(Y / l_y * 8 * jnp.pi)
)


def prepare_plot():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    cs = update_plot(0, h0, u0, v0, ax, draw=False)
    plt.colorbar(cs, label='$\\eta$ (m)')
    return fig, ax


def update_plot(t, h, u, v, ax, draw=True):
    eta = h - depth

    quiver_stride = (
        slice(1, -1, n_y // max_quivers),
        slice(1, -1, n_x // max_quivers)
    )

    ax.clear()
    cs = ax.pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        eta[1:-1, 1:-1],
        vmin=-plot_range, vmax=plot_range, cmap='RdBu_r'
    )

    if jnp.any((u[quiver_stride] != 0) | (v[quiver_stride] != 0)):
        ax.quiver(
            x[quiver_stride[1]] / 1e3,
            y[quiver_stride[0]] / 1e3,
            u[quiver_stride],
            v[quiver_stride],
            clip_on=False
        )

    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (km)')
    ax.set_ylabel('$y$ (km)')
    ax.set_xlim(x[1] / 1e3, x[-2] / 1e3)
    ax.set_ylim(y[1] / 1e3, y[-2] / 1e3)
    ax.set_title(
        't=%5.2f days, R=%5.1f km, c=%5.1f m/s '
        % (t / 86400, rossby_radius / 1e3, phase_speed)
    )

    if draw:
        plt.pause(0.1)

    return cs


def enforce_boundaries(arr, grid):
    assert grid in ("h", "u", "v")
    if periodic_boundary_x:
        arr = arr.at[:, 0].set(arr[:, -2])
        arr = arr.at[:, -1].set(arr[:, 1])
    elif grid == "u":
        arr = arr.at[:, -2].set(0.0)
    if grid == "v":
        arr = arr.at[-2, :].set(0.0)
    return arr


def iterate_shallow_water():
    # allocate arrays
    u, v, h = jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x))
    du, dv, dh = jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x))
    du_new, dv_new, dh_new = jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x))
    fe, fn = jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x))
    q, ke = jnp.empty((n_y, n_x)), jnp.empty((n_y, n_x))

    # initial conditions
    h = h0
    u = u0
    v = v0

    # boundary values of h must not be used
    h = h.at[0, :].set(jnp.nan)
    h = h.at[-1, :].set(jnp.nan)
    h = h.at[:, 0].set(jnp.nan)
    h = h.at[:, -1].set(jnp.nan)

    h = enforce_boundaries(h, 'h')
    u = enforce_boundaries(u, 'u')
    v = enforce_boundaries(v, 'v')

    first_step = True

    # time step equations
    while True:
        hc = jnp.pad(h[1:-1, 1:-1], 1, 'edge')
        hc = enforce_boundaries(hc, 'h')

        fe = fe.at[1:-1, 1:-1].set(0.5 * (hc[1:-1, 1:-1] + hc[1:-1, 2:]) * u[1:-1, 1:-1])
        fn = fn.at[1:-1, 1:-1].set(0.5 * (hc[1:-1, 1:-1] + hc[2:, 1:-1]) * v[1:-1, 1:-1])
        fe = enforce_boundaries(fe, 'u')
        fn = enforce_boundaries(fn, 'v')

        dh_new = dh_new.at[1:-1, 1:-1].set(-(
            (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
            + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
        ))

        if linear_momentum_equation:
            v_avg = 0.25 * (v[1:-1, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[:-2, 2:])
            du_new = du_new.at[1:-1, 1:-1].set(
                coriolis_param[1:-1, 1:-1] * v_avg - gravity * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx
            )
            u_avg = 0.25 * (u[1:-1, 1:-1] + u[1:-1, :-2] + u[2:, 1:-1] + u[2:, :-2])
            dv_new = dv_new.at[1:-1, 1:-1].set(
                -coriolis_param[1:-1, 1:-1] * u_avg - gravity * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy
            )

        else:  # nonlinear momentum equation
            # planetary and relative vorticity
            q = q.at[1:-1, 1:-1].set(coriolis_param[1:-1, 1:-1] + (
                (v[1:-1, 2:] - v[1:-1, 1:-1]) / dx
                - (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy
            ))
            # potential vorticity
            q = q.at[1:-1, 1:-1].set(q[1:-1, 1:-1] * 1. / (
                0.25 * (hc[1:-1, 1:-1] + hc[1:-1, 2:] + hc[2:, 1:-1] + hc[2:, 2:])
            ))
            q = enforce_boundaries(q, 'h')

            du_new = du_new.at[1:-1, 1:-1].set(
                -gravity * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx
                + 0.5 * (
                    q[1:-1, 1:-1] * 0.5 * (fn[1:-1, 1:-1] + fn[1:-1, 2:])
                    + q[:-2, 1:-1] * 0.5 * (fn[:-2, 1:-1] + fn[:-2, 2:])
                )
            )
            dv_new = dv_new.at[1:-1, 1:-1].set(
                -gravity * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy
                - 0.5 * (
                    q[1:-1, 1:-1] * 0.5 * (fe[1:-1, 1:-1] + fe[2:, 1:-1])
                    + q[1:-1, :-2] * 0.5 * (fe[1:-1, :-2] + fe[2:, :-2])
                )
            )
            ke = ke.at[1:-1, 1:-1].set(0.5 * (
                0.5 * (u[1:-1, 1:-1] ** 2 + u[1:-1, :-2] ** 2)
                + 0.5 * (v[1:-1, 1:-1] ** 2 + v[:-2, 1:-1] ** 2)
            ))
            ke = enforce_boundaries(ke, 'h')

            du_new = du_new.at[1:-1, 1:-1].set(du_new[1:-1, 1:-1] - (ke[1:-1, 2:] - ke[1:-1, 1:-1]) / dx)
            dv_new = dv_new.at[1:-1, 1:-1].set(dv_new[1:-1, 1:-1] - (ke[2:, 1:-1] - ke[1:-1, 1:-1]) / dy)

        if first_step:
            u = u.at[1:-1, 1:-1].set(u[1:-1, 1:-1] + dt * du_new[1:-1, 1:-1])
            v = v.at[1:-1, 1:-1].set(v[1:-1, 1:-1] + dt * dv_new[1:-1, 1:-1])
            h = h.at[1:-1, 1:-1].set(h[1:-1, 1:-1] + dt * dh_new[1:-1, 1:-1])
            first_step = False
        else:
            u = u.at[1:-1, 1:-1].set(u[1:-1, 1:-1] + dt * (
                adams_bashforth_a * du_new[1:-1, 1:-1]
                + adams_bashforth_b * du[1:-1, 1:-1]
            ))
            v = v.at[1:-1, 1:-1].set(v[1:-1, 1:-1] + dt * (
                adams_bashforth_a * dv_new[1:-1, 1:-1]
                + adams_bashforth_b * dv[1:-1, 1:-1]
            ))
            h = h.at[1:-1, 1:-1].set(h[1:-1, 1:-1] + dt * (
                adams_bashforth_a * dh_new[1:-1, 1:-1]
                + adams_bashforth_b * dh[1:-1, 1:-1]
            ))

        h = enforce_boundaries(h, 'h')
        u = enforce_boundaries(u, 'u')
        v = enforce_boundaries(v, 'v')

        if lateral_viscosity > 0:
            # lateral friction
            fe = fe.at[1:-1, 1:-1].set(lateral_viscosity * (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx)
            fn = fn.at[1:-1, 1:-1].set(lateral_viscosity * (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy)
            fe = enforce_boundaries(fe, "u")
            fn = enforce_boundaries(fn, "v")

            u = u.at[1:-1, 1:-1].set(
                u[1:-1, 1:-1]
                + dt
                * (
                    (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
                    + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
                )
            )

            fe = fe.at[1:-1, 1:-1].set(lateral_viscosity * (v[1:-1, 2:] - u[1:-1, 1:-1]) / dx)
            fn = fn.at[1:-1, 1:-1].set(lateral_viscosity * (v[2:, 1:-1] - u[1:-1, 1:-1]) / dy)
            fe = enforce_boundaries(fe, "u")
            fn = enforce_boundaries(fn, "v")

            v = v.at[1:-1, 1:-1].set(
                v[1:-1, 1:-1]
                + dt
                * (
                    (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
                    + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
                )
            )

        # rotate quantities
        du = du.at[:].set(du_new)
        dv = dv.at[:].set(dv_new)
        dh = dh.at[:].set(dh_new)

        yield h, u, v

if __name__ == '__main__':
    fig, ax = prepare_plot()
    model = iterate_shallow_water()
    for iteration, (h, u, v) in enumerate(model):
        if iteration % plot_every == 0:
            t = iteration * dt
            update_plot(t, h, u, v, ax)

        # stop if user closes plot window
        if not plt.fignum_exists(fig.number):
            break
