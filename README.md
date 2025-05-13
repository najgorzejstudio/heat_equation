# 3D Heat Diffusion Simulation Report

## 1. Introduction
This document presents a numerical simulation of the 3D heat diffusion equation using finite difference methods. The simulation tracks how heat disperses over time from an initial Gaussian temperature distribution in a cubic domain, subject to fixed boundary conditions.

## 2. Mathematical Background

The heat equation in three spatial dimensions is a partial differential equation that describes how temperature evolves over space and time in a medium. The general form of the heat equation is:

```
∂u/∂t = α ∇²u
```

where `u(x, y, z, t)` is the temperature field, `α` is the thermal diffusivity (m²/s), and `∇²` is the Laplacian operator. In Cartesian coordinates, the Laplacian expands to:

```
∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
```

### 2.1 Physical Interpretation

This equation models the physical process of heat diffusion: heat flows from high to low temperature regions, smoothing out temperature differences over time. The thermal diffusivity `α` controls how quickly heat spreads in the medium.

### 2.2 Discretization

To simulate the heat equation numerically, the continuous domain is discretized into a regular grid. The time derivative is approximated using a forward finite difference, and spatial derivatives use central finite differences. The discretized form of the heat equation becomes:

```
uᶰ⁺¹[i,j,k] = uⁿ[i,j,k] + α·dt·[(uⁿ[i+1,j,k] - 2uⁿ[i,j,k] + uⁿ[i-1,j,k])/dx² +
                                (uⁿ[i,j+1,k] - 2uⁿ[i,j,k] + uⁿ[i,j-1,k])/dy² +
                                (uⁿ[i,j,k+1] - 2uⁿ[i,j,k] + uⁿ[i,j,k-1])/dz²]
```

### 2.3 Stability Condition

The explicit method used for time stepping requires a small enough time step to ensure stability. The condition for stability in 3D is:

```
dt ≤ h² / (6α)
```

This restriction ensures that the simulation does not diverge numerically. It is derived from the Courant–Friedrichs–Lewy (CFL) condition.

## 3. Simulation Overview

The simulation initializes a temperature field with a broad Gaussian centered in the domain, representing a heat source. Boundary temperatures are fixed to an ambient value. Over time, heat diffuses from the center toward the boundaries.

## 4. Visualization

The temperature field is visualized using Plotly's 3D isosurface plots. These show surfaces of constant temperature, providing insight into how the heat spreads. The colors map to temperature values using a blue-red (coolwarm) color scale.
