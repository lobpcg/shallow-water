# Shallow-water modelling in Python

This repository master branch is forked from the currently archived read-only project https://github.com/dionhaefner/shallow-water

It contains two implementations by https://github.com/dionhaefner of the shallow-water equations that are suitable to study a wide range of wave and ocean circulation phenomena, including non-linear effects.

They are a product of the [Bornö summer school 2018](https://nbiocean.bitbucket.io/bornoe2018b/), led by [Markus Jochum](https://www.nbi.ku.dk/english/staff/?pure=en/persons/437464) and [Carsten Eden](https://www.ifm.uni-hamburg.de/en/institute/staff/eden.html).

A preview of the non-linear setup:

![Nonlinear model spin-up](preview.gif?raw=true)

## Features

### Simple (linear) implementation

- Mass conserving on (Cartesian) Arakawa C-grid
- Mixed-time discretization
- Coriolis force on an f-plane
- Conditionally stable for `Δt <= √2 / f`

### Fully non-linear implementation

All features of the simple implementation, plus...

- Adams-Bashforth time stepping scheme
- Lateral friction
- Varying Coriolis parameter (β-plane)
- Fully non-linear momentum and continuity equations
- Energy conserving scheme by Sadourny (1975)
- Rigid or periodic boundary conditions


# Shallow-water modelling in JAX

The new branch additionally contains JAX versions of the two pure Python modules and pytest functions checking that the new JAX and original Python implementations result in the same for the linear model and approximately the same for the nonlinear model outputs.