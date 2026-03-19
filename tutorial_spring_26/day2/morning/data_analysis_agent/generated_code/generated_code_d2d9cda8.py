import sys
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def install_packages(required_packages):
    if not required_packages:
        return
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-t", os.environ["TMPDIR"]] + required_packages, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages with uv: {e}")
        sys.exit(1)


def main():
    required_packages = ["numpy", "matplotlib", "scipy"]

    if required_packages:
        install_packages(required_packages)

    # All import statements other than sys and subprocess go here
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Simulation parameters
    dt = 0.01  # Time step
    total_time = 20  # Total simulation time (1 period)
    num_steps = int(total_time / dt)
    
    # Vortex parameters
    gamma = 1.0  # Circulation (m^2/s)
    r0 = 1.0  # Initial separation distance
    
    # Initial positions of vortices (at y = +/- r0/2, x = 0)
    # Let's place them at (0, 0.5) and (0, -0.5)
    p1_initial = np.array([0.0, r0 / 2.0])
    p2_initial = np.array([0.0, -r0 / 2.0])
    
    # For a pair of equal vortices with same sign (positive)
    # The vortices orbit around the center with angular velocity omega = gamma/(4*pi*r)
    omega = gamma / (4 * np.pi * r0)
    
    print(f"Vortex orbital period: {2*np.pi/omega:.2f} seconds")
    
    # Equations of motion for point vortices
    def equations_of_motion(t, y):
        """
        Return derivatives of positions of two vortices.
        y = [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = y
        
        # Compute distance between the vortices
        dx = x2 - x1
        dy = y2 - y1
        r_squared = dx**2 + dy**2
        r = np.sqrt(r_squared)
        
        # Avoid division by zero
        if r < 1e-10:
            return [0, 0, 0, 0]
        
        # Velocity due to second vortex on first (opposite sign for same sign vortices in potential flow)
        # v = gamma/(2*pi*r) * (-dy, dx)/r
        # For same-sign vortices in a potential flow system like this, they orbit each other
        # The induced velocity of vortex 2 on vortex 1 is 
        # v1_x = gamma/(4*pi) * (y2 - y1)/r^2
        # v1_y = -gamma/(4*pi) * (x2 - x1)/r^2
        
        factor1 = gamma / (4 * np.pi * r_squared)
        dx1_dt = factor1 * dy
        dy1_dt = -factor1 * dx
        
        factor2 = gamma / (4 * np.pi * r_squared)
        dx2_dt = -factor2 * dy
        dy2_dt = factor2 * dx
        
        return [dx1_dt, dy1_dt, dx2_dt, dy2_dt]
    
    # Set up initial condition for the integration
    y0 = np.concatenate([p1_initial, p2_initial])
    
    # Define time span (from 0 to total_time)
    t_span = (0.0, total_time)
    
    # Integrate equations of motion using scipy.integrate.solve_ivp
    solution = solve_ivp(equations_of_motion, t_span, y0, method='RK45', t_eval=np.linspace(0, total_time, num_steps),
                         rtol=1e-8, atol=1e-8)
    
    # Extract time array and solution
    times = solution.t
    positions = solution.y  # shape (4, num_steps)
    
    # Extract vortex positions
    x1s = positions[0]
    y1s = positions[1]
    x2s = positions[2]
    y2s = positions[3]
    
    # Create output directory
    os.makedirs('./generated_images/fluid_flow', exist_ok=True)
    
    # Compute and save 10 flowfield snapshots at regular intervals
    num_snapshots = 10
    steps_per_snapshot = len(times) // num_snapshots
    
    # Define grid for plotting the flow field
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    for i in range(num_snapshots):
        # Get current time
        step = i * steps_per_snapshot
        current_time = times[step]
        
        # Get vortex positions
        p1_x = x1s[step]
        p1_y = y1s[step]
        p2_x = x2s[step]
        p2_y = y2s[step]
        
        # Compute stream function and velocity field
        # Using the potential flow solution for two point vortices
        # For a vortex at (x0, y0) with circulation gamma:
        # psi = -gamma/(2*pi) * arctan((y-y0)/(x-x0))
        # u = -partial_y(psi), v = partial_x(psi)
        
        # Velocity field at each grid point due to both vortices
        # We'll compute the stream function and plot the flow field
        
        # Stream function contributions from both vortices
        psi1 = -gamma / (2 * np.pi) * np.arctan2(Y - p1_y, X - p1_x)
        psi2 = -gamma / (2 * np.pi) * np.arctan2(Y - p2_y, X - p2_x)
        
        # Total stream function
        psi_total = psi1 + psi2
        
        # Create and save plot
        plt.figure(figsize=(8, 6))
        
        # Plot streamlines
        levels = np.linspace(psi_total.min(), psi_total.max(), 30)
        plt.contour(X, Y, psi_total, levels=levels, colors='blue', linewidths=1.2)
        
        # Plot vortex positions
        plt.scatter([p1_x, p2_x], [p1_y, p2_y], s=50, c='red', marker='o')
        
        # Add labels (time and vortices)
        plt.title(f"Time = {current_time:.2f} s")
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        
        # Save plot
        filename = f'./generated_images/fluid_flow/snapsht_{i:02d}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
    print(f"Saved {num_snapshots} snapshots to ./generated_images/fluid_flow/")

if __name__ == "__main__":
    main()