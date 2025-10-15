import math
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os
# Set a consistent random seed for reproducibility
np.random.seed(42)

def get_cross_sections(energy, region):
    """
    Returns the macroscopic cross sections for a given energy and region.
    The energy is mapped to one of ten energy groups.
    Units are 1/cm.
    
    Args:
        energy (float): Neutron energy in MeV.
        region (int): The material region (0: fuel, 1: cladding, 2: moderator).

    Returns:
        list: [sigma_f, sigma_c, sigma_s, sigma_t]
    """
    
    # Macroscopic cross section data
    sigma_f = np.array([[1.05e-1, 0, 0],
                        [5.96e-2, 0, 0],
                        [6.02e-2, 0, 0],
                        [1.06e-1, 0, 0],
                        [2.46e-1, 0, 0],
                        [2.50e-1, 0, 0],
                        [1.07e-1, 0, 0],
                        [1.28e+0, 0, 0],
                        [9.30e+0, 0, 0],
                        [2.58e+1, 0, 0]])
    sigma_c = np.array([[1.41e-6, 1.71e-2, 3.34e-6],
                        [1.34e-3, 7.83e-3, 3.34e-6],
                        [1.10e-2, 2.83e-4, 2.56e-7],
                        [3.29e-2, 4.52e-6, 6.63e-7],
                        [8.23e-2, 1.06e-5, 2.24e-7],
                        [4.28e-2, 4.39e-6, 1.27e-7],
                        [9.90e-2, 1.25e-5, 2.02e-7],
                        [2.51e-1, 3.98e-5, 6.02e-7],
                        [2.12e+0, 1.26e-4, 1.84e-6],
                        [4.30e+0, 3.95e-4, 5.76e-6]])
    sigma_s = np.array([[2.76e-1, 1.44e-1, 1.27e-2],
                        [3.88e-1, 1.76e-1, 7.36e-2],
                        [4.77e-1, 3.44e-1, 2.65e-1],
                        [6.88e-1, 2.66e-1, 5.72e-1],
                        [9.38e-1, 2.06e-1, 6.69e-1],
                        [1.52e+0, 2.14e-1, 6.81e-1],
                        [2.30e+0, 2.23e-1, 6.82e-1],
                        [2.45e+0, 2.31e-1, 6.83e-1],
                        [9.79e+0, 2.40e-1, 6.86e-1],
                        [4.36e+1, 2.41e-1, 6.91e-1]])
    sigma_t = sigma_f + sigma_c + sigma_s

    # Energy group boundaries (MeV)
    group_energy_boundaries = [3e+1, 3e+0, 3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7, 3e-8]
    
    group = 9 # Default to the lowest group if energy is below all boundaries
    for g in range(len(group_energy_boundaries)):
        if energy >= group_energy_boundaries[g]:
            group = g
            break
            
    sig_f = sigma_f[group][region]
    sig_c = sigma_c[group][region]
    sig_s = sigma_s[group][region]
    sig_t = sigma_t[group][region]
    
    print(f"  Region: {region}, Group: {group}, E: {energy:.4e} MeV -> sig_f: {sig_f:.4e}, sig_c: {sig_c:.4e}, sig_s: {sig_s:.4e}, sig_t: {sig_t:.4e}")
    return [sig_f, sig_c, sig_s, sig_t]

def get_group(energy):
    """Returns the energy group number for a given energy."""
    group_energy_boundaries = [3e+1, 3e+0, 3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7, 3e-8]
    group = 9
    for g in range(len(group_energy_boundaries)):
        if energy >= group_energy_boundaries[g]:
            group = g
            break
    return group

def get_material_data(region):
    """Returns material properties for a given region."""
    if region == 0:  # Fuel
        return {"name": "fuel(UO2)", "A": 238.02891, "density": 10.97}
    elif region == 1:  # Cladding
        return {"name": "cladding(Aluminium)", "A": 26.981539, "density": 2.70}
    elif region == 2:  # Moderator
        return {"name": "moderator(water)", "A": 1.00794, "density": 1}

def animate_neutron_paths(neutron_paths, interaction_points, geometry_params, filename="neutron_animation.gif"):
    """
    Create an animation of neutron trajectories.
    
    Args:
        neutron_paths (list): List of neutron paths, each path is a list of (x, y) coordinates
        interaction_points (list): List of (x, y) coordinates for interaction points
        geometry_params (dict): Geometry parameters (r_fuel, r_clad_out, pitch)
        filename (str): Output filename for the animation
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw geometry
    r_fuel, r_clad_out, pitch = geometry_params['r_fuel'], geometry_params['r_clad_out'], geometry_params['pitch']
    ax.add_artist(plt.Circle((0,0), r_fuel, fill=False, color='green', label='Fuel', linewidth=2))
    ax.add_artist(plt.Circle((0,0), r_clad_out, fill=False, color='red', label='Cladding', linewidth=2))
    ax.add_artist(plt.Rectangle((-pitch/2, -pitch/2), width=pitch, height=pitch, fill=False, color='blue', label='Moderator', linewidth=2))
    
    ax.set_xlim(-pitch/2, pitch/2)
    ax.set_ylim(-pitch/2, pitch/2)
    ax.set_title("Neutron Transport Animation", fontsize=14)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Initialize empty line collections and scatter plots
    lines = []
    scatters = []
    
    # Prepare data for animation
    max_path_length = max(len(path) for path in neutron_paths) if neutron_paths else 0
    
    def animate(frame):
        # Clear previous frame
        #for line in lines:
            #line.remove()
        #for scatter in scatters:
            #scatter.remove()
        #lines.clear()
        #scatters.clear()
        
        # Draw paths up to current frame
        for i, path in enumerate(neutron_paths):
            if frame < len(path):
                # Current position
                current_x, current_y = path[frame]
                
                # Draw path up to current position
                if frame > 0:
                    path_segment = path[:frame+1]
                    x_vals = [p[0] for p in path_segment]
                    y_vals = [p[1] for p in path_segment]
                    line, = ax.plot(x_vals, y_vals, '-', alpha=0.7, linewidth=1.5, color=plt.cm.tab10(i % 10))
                    lines.append(line)
                
                # Draw current position as a point
                scatter = ax.scatter([current_x], [current_y], c=[plt.cm.tab10(i % 10)], s=50, alpha=0.8)
                scatters.append(scatter)
        
        # Draw interaction points that have occurred up to this frame
        interaction_x = [p[0] for p in interaction_points if p[2] <= frame]
        interaction_y = [p[1] for p in interaction_points if p[2] <= frame]
        if interaction_x:
            scatter_int = ax.scatter(interaction_x, interaction_y, c='red', s=80, marker='*', alpha=0.8, label='Interactions')
            scatters.append(scatter_int)
        
        ax.set_title(f"Neutron Transport Animation - Frame {frame}/{max_path_length}", fontsize=14)
        frame_dir = "frames"
        os.makedirs(frame_dir, exist_ok=True)
        frame_filename = os.path.join(frame_dir, f"frame_{frame:04d}.png")
        fig.canvas.draw()  # Frame'i tam olarak çiz
        fig.savefig(frame_filename, dpi=150)
        #plt.savefig(frame_filename, dpi=150)
        print(f"Saved {frame_filename}")
        return lines + scatters
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=max_path_length, interval=200, blit=False, repeat=False, save_count=max_path_length)

    # Save animation
    print(f"Saving animation to {filename}...")
    anim.save(filename, writer='pillow', fps=5, dpi=100)
    # print("Animation saved!")
    #plt.show() 
    # plt.close(fig)
    return anim

def qualifying_monte_carlo(num_neutrons, create_animation=True):
    """
    Monte Carlo simulation of neutron transport in a unit cell.
    
    Args:
        num_neutrons (int): The number of neutrons to simulate.
        create_animation (bool): Whether to create an animation of neutron paths.
    """
    
    print(f"Number of Neutrons.......................= {num_neutrons}")
    
    # --- Geometry Parameters ---
    r_fuel = 0.53       # Fuel radius (cm)
    r_clad_in = 0.53    # Cladding inner radius (cm)
    r_clad_out = 0.90   # Cladding outer radius (cm)
    pitch = 1.837       # Cell pitch (cm)

    # --- Simulation Counters ---
    fission_count = 0
    capture_count = 0
    scattering_count = 0
    leakage_count = 0
    neutrons_produced = 0
    
    # Lists for tracking data
    interaction_points = []  # (x, y, frame)
    neutron_paths = []  # List of paths for each neutron
    frame_counter = 0
    
    # --- Neutron Initial State Generation ---
    # Maxwelian Neutron Energy Distribution [MeV]
    neutron_energies = maxwell.rvs(size=num_neutrons) 
    
    # Power-law distribution for radial position to simulate uniform fission density
    rand_r = np.random.power(2, size=num_neutrons) 
    radii = rand_r * r_fuel 
    
    # Uniform angular distribution
    thetas_initial = 2 * np.pi * np.random.random(size=num_neutrons)
    
    initial_x = radii * np.cos(thetas_initial)
    initial_y = radii * np.sin(thetas_initial)
    
    # --- Energy Group Flux Tallies ---
    fuel_surf_neu_num = np.zeros(10)
    clad_surf_neu_num = np.zeros(10)
    
    # Plot setup for trajectories
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.add_artist(plt.Circle((0,0), r_fuel, fill=False, color='green', label='Fuel', linewidth=2))
    ax.add_artist(plt.Circle((0,0), r_clad_out, fill=False, color='red', label='Cladding', linewidth=2))
    ax.add_artist(plt.Rectangle((-pitch/2, -pitch/2), width=pitch, height=pitch, fill=False, color='blue', label='Moderator', linewidth=2))
    plt.xlim(-pitch/2, pitch/2)
    plt.ylim(-pitch/2, pitch/2)
    plt.title("Neutron Trajectories in a Fuel Cell", fontsize=14)
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.grid(True, alpha=0.3)
    
    # Main simulation loop for each neutron
    for i in range(num_neutrons):
        print(f"\n--- Starting Neutron {i+1} ---")
        
        # Initialize neutron state
        alive = True
        region = 0  # Starts in the fuel region
        x, y = initial_x[i], initial_y[i]
        energy = neutron_energies[i]
        
        # Initial direction is random
        theta = 2 * np.pi * np.random.random()
        
        path = [(x, y)]  # Store all positions for this neutron
        
        while alive:
            frame_counter += 1
            
            # Get cross sections for the current state
            sig_f, sig_c, sig_s, sig_t = get_cross_sections(energy, region)
            
            # Distance to the next interaction
            d_interaction = - (1 / sig_t) * np.log(np.random.random())
            
            d_boundary = float('inf')
            next_region = region
            
            # --- Geometry Logic to find minimum distance to a boundary ---
            
            # Current squared radius
            r_sq = x**2 + y**2
            
            if region == 0:  # Fuel region
                # Distance to the outer fuel surface (r_clad_in)
                a = 1
                b = 2 * (x * np.cos(theta) + y * np.sin(theta))
                c = r_sq - r_clad_in**2
                delta = b**2 - 4*a*c
                
                if delta >= 0:
                    d_temp = (-b + np.sqrt(delta)) / (2*a)
                    if d_temp > 1e-9:
                        d_boundary = d_temp
                        next_region = 1
                        
            elif region == 1:  # Cladding region
                # Distance to the inner cladding surface (r_clad_in)
                a = 1
                b = 2 * (x * np.cos(theta) + y * np.sin(theta))
                c_in = r_sq - r_clad_in**2
                delta_in = b**2 - 4*a*c_in
                if delta_in >= 0:
                    d_temp = (-b - np.sqrt(delta_in)) / (2*a)
                    if d_temp > 1e-9:
                        d_boundary = d_temp
                        next_region = 0

                # Distance to the outer cladding surface (r_clad_out)
                c_out = r_sq - r_clad_out**2
                delta_out = b**2 - 4*a*c_out
                if delta_out >= 0:
                    d_temp = (-b + np.sqrt(delta_out)) / (2*a)
                    if d_temp > 1e-9 and d_temp < d_boundary:
                        d_boundary = d_temp
                        next_region = 2

            elif region == 2:  # Moderator region
                # Distance to the inner cladding surface (r_clad_out)
                a = 1
                b = 2 * (x * np.cos(theta) + y * np.sin(theta))
                c_out = r_sq - r_clad_out**2
                delta_out = b**2 - 4*a*c_out
                if delta_out >= 0:
                    d_temp = (-b - np.sqrt(delta_out)) / (2*a)
                    if d_temp > 1e-9:
                        d_boundary = d_temp
                        next_region = 1

                # Distance to the square cell boundaries
                distances_to_square = []
                if np.cos(theta) > 1e-9: distances_to_square.append((pitch/2 - x) / np.cos(theta))
                if np.cos(theta) < -1e-9: distances_to_square.append((-pitch/2 - x) / np.cos(theta))
                if np.sin(theta) > 1e-9: distances_to_square.append((pitch/2 - y) / np.sin(theta))
                if np.sin(theta) < -1e-9: distances_to_square.append((-pitch/2 - y) / np.sin(theta))

                for d_sq in distances_to_square:
                    if d_sq > 1e-9 and d_sq < d_boundary:
                        d_boundary = d_sq
                        next_region = -1 # Special code for leakage/wrap-around
            
            # --- Check if interaction or boundary crossing happens first ---
            
            if d_interaction < d_boundary:
                # Interaction happens inside the current region
                print(f"  Interaction in {get_material_data(region)['name']}")
                
                x += d_interaction * np.cos(theta)
                y += d_interaction * np.sin(theta)
                path.append((x, y))
                
                interaction_points.append((x, y, frame_counter))
                
                # Determine interaction type based on cross sections
                rnd = np.random.random()
                
                if rnd <= sig_f / sig_t:
                    print("  Event: Fission")
                    fission_count += 1
                    # Neutrons from fission
                    num_fission_neutrons = 2 if np.random.random() < 0.5 else 3
                    neutrons_produced += num_fission_neutrons
                    alive = False
                    
                elif rnd <= (sig_f + sig_c) / sig_t:
                    print("  Event: Capture")
                    capture_count += 1
                    alive = False
                    
                else:
                    print("  Event: Scattering")
                    scattering_count += 1
                    
                    # Update energy after scattering
                    material_A = get_material_data(region)['A']
                    if material_A > 1:
                        ksi = 1 + np.log((material_A - 1) / (material_A + 1)) * (material_A - 1)**2 / (2 * material_A)
                    else: # For Hydrogen (A~1), ksi is approximately 1
                        ksi = 1
                    energy *= np.exp(-ksi * np.random.random()) # Random energy loss
                    
                    # New random direction
                    theta = 2 * np.pi * np.random.random()
                    
            else:
                # Boundary crossing happens first
                
                x += d_boundary * np.cos(theta)
                y += d_boundary * np.sin(theta)
                path.append((x, y))
                
                # Update flux tallies
                group = get_group(energy)
                if region == 0:
                    fuel_surf_neu_num[group] += 1
                elif region == 1 and next_region == 2:
                    clad_surf_neu_num[group] += 1
                    
                print(f"  Boundary crossing from Region {region} to {next_region}")
                
                # Handle wrap-around for moderator region
                if next_region == -1:
                    leakage_count += 1
                    
                    # Periodic boundary conditions
                    if abs(x) > pitch / 2 - 1e-9: x = -x
                    if abs(y) > pitch / 2 - 1e-9: y = -y
                    print("  Event: Leakage (periodic boundary condition applied)")
                    
                    # Neutron direction is maintained after crossing boundary
                    region = 2 # Re-enter moderator region
                    
                else:
                    region = next_region
                    
        # Store the complete path for this neutron
        neutron_paths.append(path)
        
        # Plot the neutron's path
        x_vals = [p[0] for p in path]
        y_vals = [p[1] for p in path]
        plt.plot(x_vals, y_vals, '-', alpha=0.7, linewidth=1.5, color=plt.cm.tab10(i % 10))
        plt.scatter(x_vals[-1], y_vals[-1], color=plt.cm.tab10(i % 10), s=30)

    # Plot interaction points
    if interaction_points:
        int_x = [p[0] for p in interaction_points]
        int_y = [p[1] for p in interaction_points]
        plt.scatter(int_x, int_y, c='red', marker='*', s=100, label='Interactions', zorder=5)
    
    # plt.legend()
    # plt.savefig("neutron_paths.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # Create animation if requested
    if create_animation and neutron_paths:
        geometry_params = {
            'r_fuel': r_fuel,
            'r_clad_out': r_clad_out,
            'pitch': pitch
        }
        animate_neutron_paths(neutron_paths, interaction_points, geometry_params)

    # --- Results and Final Plotting ---
    absorption_count = fission_count + capture_count
    total_interactions = scattering_count + absorption_count
    
    # Effective multiplication factor (k_eff)
    if total_interactions > 0:
        keff = (neutrons_produced + leakage_count) / (absorption_count + leakage_count)
    else:
        keff = 0
        
    print("\n--- Simulation Results ---")
    print(f"Number of Neutrons.......................= {num_neutrons}")
    print(f"Total Interactions.......................= {total_interactions}")
    print(f"  Scattering Events......................= {scattering_count}")
    print(f"  Capture Events.........................= {capture_count}")
    print(f"  Fission Events.........................= {fission_count}")
    print(f"  Total Absorption Events................= {absorption_count}")
    print(f"  Leakage Events.........................= {leakage_count}")
    print(f"Neutrons Produced by Fission.............= {neutrons_produced}")
    print(f"Effective Multiplication Factor (keff)...= {keff:.4f}")
    
    # --- Flux Spectrum Plotting (Normalized) ---
    
    group_energy = [3e+1, 3e+0, 3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7, 3e-8]
    fuel_area = np.pi * r_fuel**2
    scaling_factor = 10 * num_neutrons
    
    # Simulated Flux
    flux_y = fuel_surf_neu_num / (fuel_area * scaling_factor)
    
    # Watt spectrum
    watt_energy = np.linspace(0.01, 30, 500)
    watt_flux = 0.453 * np.sinh(np.sqrt(2.29 * watt_energy)) * np.exp(-1.036 * watt_energy)
    
    # Normalization (Units are being created under the field)
    flux_y_norm = flux_y / np.trapz(flux_y, group_energy)
    watt_flux_norm = watt_flux / np.trapz(watt_flux, watt_energy)
    
    # --- Plotting ---
    # plt.figure()
    # plt.plot(group_energy, -flux_y_norm, '-bo', label='Normalized Simulated Flux')
    # plt.plot(watt_energy, watt_flux_norm, '-r', label='Normalized Watt Spectrum')
    # plt.xlabel("Energy (MeV)")
    # plt.ylabel("Normalized Flux")
    # plt.title("Normalized Neutron Flux Spectrum")
    # plt.legend(loc='upper right')
    # plt.grid(True, which="both", linestyle='--')
    # #plt.xscale('log')
    # plt.savefig("flux_spectrum.png", dpi=300)
    # plt.show()

if __name__ == '__main__':
    qualifying_monte_carlo(50, create_animation=True)