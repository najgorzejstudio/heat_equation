import numpy as np
import plotly.graph_objects as go
import time

# --- Simulation Parameters ---
L = 2.0
Nx = Ny = Nz = 20
alpha = 0.1
T_final = 5.0
num_frames = 60

# --- Environment and Initial State ---
T_ambient = 0.0
T_initial_peak = 100.0

# --- Discretization ---
dx = L / (Nx - 1)
dy = L / (Ny - 1)
dz = L / (Nz - 1)
h = dx

# --- Time Step Calculation & Stability Check ---
stability_limit = h**2 / (6 * alpha)
dt_target = T_final / (num_frames * 10)
dt = min(dt_target, stability_limit * 0.9)
if dt < dt_target: print(f"Warning: Target dt > stability limit. Using dt={dt:.6f}")
else: print(f"Using dt = {dt:.6f}")
Nt = int(T_final / dt)
frame_interval_steps = max(1, Nt // num_frames)
actual_num_frames = Nt // frame_interval_steps

print(f"--- Parameters Summary ---")
print(f"Grid size: {Nx}x{Ny}x{Nz}")
print(f"Total time: {T_final}, Steps: {Nt}, dt: {dt:.6f}")
print(f"Frames: {actual_num_frames}, Steps/Frame: {frame_interval_steps}")
print("-" * 20)

# --- Grid Setup ---
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
z = np.linspace(0, L, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- Initial Condition ---
u = np.full((Nx, Ny, Nz), T_ambient)
center_x, center_y, center_z = L / 2, L / 2, L / 2

# ---> Set sigma = L <---
sigma = L
print(f"NOTE: Using sigma = L = {L:.2f}. Initial heat distribution will be very broad.")

delta_T = T_initial_peak - T_ambient
# Calculate initial temperature distribution
u += delta_T * np.exp(-((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2) / (2 * sigma**2))

# IMPORTANT: Because sigma=L, the initial Gaussian has non-negligible values at boundaries.
# We *still* want the boundaries fixed at T_ambient for the simulation. So, enforce it *after* calculating the initial state.
u[0, :, :] = T_initial_peak; u[-1, :, :] = T_initial_peak
u[:, 0, :] = T_ambient; u[:, -1, :] = T_ambient
u[:, :, 0] = T_ambient; u[:, :, -1] = T_ambient

u_new = u.copy()

# --- Simulation Loop & Frame Storage ---
print(f"Running simulation...")
start_sim_time = time.time()
frames_data = []
time_points = []
frames_data.append(u.copy()) # Store the state *after* boundary enforcement
time_points.append(0.0)

for n in range(Nt):
    laplacian_x = (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / dx**2
    laplacian_y = (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / dy**2
    laplacian_z = (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / dz**2
    u_new[1:-1, 1:-1, 1:-1] = u[1:-1, 1:-1, 1:-1] + alpha * dt * (laplacian_x + laplacian_y + laplacian_z)

    u = u_new.copy()
    if (n + 1) % frame_interval_steps == 0:
        current_time = (n + 1) * dt
        if current_time <= T_final * 1.01:
             frames_data.append(u.copy())
             time_points.append(current_time)
        if np.any(np.isnan(u)): print(f"ERROR: Simulation diverged!"); break

end_sim_time = time.time()
print(f"Simulation finished in {end_sim_time - start_sim_time:.2f} sec.")
print(f"Generated {len(frames_data)} frames.")

# --- Plotly Visualization ---
print("Creating Plotly figure with dynamic isosurfaces and coolwarm scale...")

# Determine GLOBAL temperature range for COLORBAR
initial_u = frames_data[0] # Use the corrected initial state
# Find the actual peak *after* boundary enforcement, might be slightly lower than T_initial_peak if center is close to boundary
max_temp_global = initial_u.max()
min_temp_global = T_ambient # The lowest temperature will be ambient

print(f"Global Temp Range for Colorbar: {min_temp_global:.2f} to {max_temp_global:.2f}")

# Calculate initial isosurface display range (relative) for the first frame
current_max_initial = initial_u.max()
delta_T_initial = max(1e-6, current_max_initial - T_ambient)
isomin_initial = T_ambient + 0.1 * delta_T_initial # Show from 10% level
isomax_initial = T_ambient + 0.9 * delta_T_initial # Show up to 90% level

print(f"Initial Isosurface range: {isomin_initial:.2f} to {isomax_initial:.2f}")

# Create the figure with the initial trace
fig = go.Figure(
    data=[
        go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=initial_u.flatten(),
            isomin=isomin_initial,
            isomax=isomax_initial,
            surface_count=10,
            opacity=0.4,
            caps=dict(x_show=False, y_show=False, z_show=False),
            # ---> Use coolwarm colorscale <---
            colorscale='bluered',
            cmin=min_temp_global,    # GLOBAL min for color
            cmax=max_temp_global,    # GLOBAL max for color
            colorbar=dict(title='Temperature (°C)'),
            name=f't={time_points[0]:.3f}'
        )
    ],
    layout=go.Layout(
        title=f'Heat Diffusion (Relative Isosurface, sigma=L) | t={time_points[0]:.3f}', # Updated title
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectratio=dict(x=1, y=1, z=1),
            camera_eye=dict(x=1.8, y=1.8, z=1.8),
            xaxis=dict(range=[0, L]), yaxis=dict(range=[0, L]), zaxis=dict(range=[0, L]),
        ),
        updatemenus=[dict(
            type='buttons', showactive=False,
            buttons=[dict(label='Play', method='animate',
                          args=[None, dict(frame=dict(duration=80, redraw=True),
                                           fromcurrent=True, mode='immediate')])])],
        sliders=[dict(steps=[], active=0,
                      currentvalue={"prefix": "Time: ", "suffix": " s"},
                      pad={"t": 50})]
    )
)

# --- Create frames with DYNAMIC isosurface levels ---
plotly_frames = []
slider_steps = []

for i, u_frame in enumerate(frames_data):
    current_time = time_points[i]

    # Calculate dynamic iso-range for THIS frame
    current_max = u_frame.max()
    delta_T_frame = max(1e-6, current_max - T_ambient)
    frame_isomin = T_ambient + 0.1 * delta_T_frame
    frame_isomax = T_ambient + 0.9 * delta_T_frame
    if frame_isomin >= frame_isomax: frame_isomin = T_ambient

    frame = go.Frame(
        name=f't={current_time:.3f}',
        data=[go.Isosurface(
            value=u_frame.flatten(),
            isomin=frame_isomin,     # Frame-specific isomin
            isomax=frame_isomax,     # Frame-specific isomax
            cmin=min_temp_global,    # Keep GLOBAL cmin
            cmax=max_temp_global,    # Keep GLOBAL cmax
            # ---> Use coolwarm colorscale <---
            colorscale='bluered',
            # --- Other properties (match initial trace) ---
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            surface_count=10, opacity=0.4,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )],
    )
    plotly_frames.append(frame)

    # --- Slider Step ---
    current_title = f'Heat Diffusion (Relative Isosurface, sigma=L) | t={current_time:.3f}'
    slider_step = dict(
        args=[
            [f't={current_time:.3f}'],
            dict(mode='immediate',
                 frame=dict(duration=80, redraw=True),
                 layout=dict(title=current_title))
        ],
        label=f'{current_time:.2f}',
        method='animate'
    )
    slider_steps.append(slider_step)

# Assign frames and slider steps
fig.frames = plotly_frames
fig.layout.sliders[0].steps = slider_steps

# Refine Button Args
button_args = [ None, dict(frame=dict(duration=80, redraw=True),
                          transition=dict(duration=0), fromcurrent=True, mode='immediate') ]
fig.layout.updatemenus[0].buttons[0].args = button_args

print("Showing plot...")
fig.show()
print("Plot generation complete.")
print("\nNOTE: Initial heat uses sigma=L (very broad). Isosurfaces show relative temp levels.")
print("      Colors map temperature using 'coolwarm' (Blue=Cold, Red=Hot).")