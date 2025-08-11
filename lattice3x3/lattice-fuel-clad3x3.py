import math
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell

# Set a consistent random seed for reproducibility
np.random.seed()

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
    
    # print(f"  Region: {region}, Group: {group}, E: {energy:.4e} MeV -> sig_f: {sig_f:.4e}, sig_c: {sig_c:.4e}, sig_s: {sig_s:.4e}, sig_t: {sig_t:.4e}")
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

def qualifying_monte_carlo(num_neutrons, pin, lattice, lattice_pitch):
    """
    Monte Carlo simulation of neutron transport in a unit cell.
    
    Args:
        num_neutrons (int): The number of neutrons to simulate.
        pin (dict): Dictionary defining the geometry of a single pin.
        lattice (list): 2D list defining the lattice structure.
        lattice_pitch (float): The side length of a single cell.
    """
    
    print(f"Number of Neutrons.......................= {num_neutrons}")
    
    # --- Geometry Parameters from pin and lattice objects ---
    # We need to handle different pin types within the lattice
    pin_types = [p for row in lattice for p in row]
    has_fuel = "fuel_radius" in pin_types[0] # Assumes first pin is a fuel pin
    r_fuel = pin["fuel_radius"] if has_fuel else 0
    r_clad_in = pin["clad_inner_radius"] if has_fuel else 0
    r_clad_out = pin["clad_outer_radius"]
    
    lattice_size_x = len(lattice[0])
    lattice_size_y = len(lattice)
    total_lattice_size_x = lattice_size_x * lattice_pitch
    total_lattice_size_y = lattice_size_y * lattice_pitch
    pitch = lattice_pitch
    
    # --- Simulation Counters ---
    fission_count = 0
    capture_count = 0
    scattering_count = 0
    leakage_count = 0
    neutrons_produced = 0
    
    # Lists for tracking data
    interaction_point_x = []
    interaction_point_y = []
    
    # --- Neutron Initial State Generation ---
    neutron_energies = maxwell.rvs(size=num_neutrons)
    
    rand_r = np.random.power(2, size=num_neutrons)
    # Initial position is only in fuel pins
    # A more robust solution would be to select a fuel pin at random
    radii = rand_r * r_fuel
    
    thetas_initial = 2 * np.pi * np.random.random(size=num_neutrons)
    
    initial_x_cell = radii * np.cos(thetas_initial)
    initial_y_cell = radii * np.sin(thetas_initial)
    
    # Start all neutrons in the first cell (0,0) relative to the center of the overall lattice
    initial_x = initial_x_cell + (0.5 - lattice_size_x/2) * pitch
    initial_y = initial_y_cell + (0.5 - lattice_size_y/2) * pitch
    
    # --- Energy Group Flux Tallies ---
    fuel_surf_neu_num = np.zeros(10)
    clad_surf_neu_num = np.zeros(10)
    
    # Plot setup for trajectories
    plt.figure(figsize=(10, 10))
    for i in range(lattice_size_x):
        for j in range(lattice_size_y):
            cell_center_x = (i - lattice_size_x/2 + 0.5) * pitch
            cell_center_y = (j - lattice_size_y/2 + 0.5) * pitch
            
            current_pin = lattice[j][i]
            if "fuel_radius" in current_pin:
                plt.gcf().gca().add_artist(plt.Circle((cell_center_x, cell_center_y), current_pin["fuel_radius"], fill=False, color='green', alpha=0.5))
            plt.gcf().gca().add_artist(plt.Circle((cell_center_x, cell_center_y), current_pin["clad_outer_radius"], fill=False, color='red', alpha=0.5))
            plt.gcf().gca().add_artist(plt.Rectangle((cell_center_x - pitch/2, cell_center_y - pitch/2), width=pitch, height=pitch, fill=False, color='blue', alpha=0.5))

    plt.xlim(-total_lattice_size_x/2, total_lattice_size_x/2)
    plt.ylim(-total_lattice_size_y/2, total_lattice_size_y/2)
    plt.title(f"Neutron Trajectories in a {lattice_size_x}x{lattice_size_y} Fuel Cell Lattice")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    
    # Add a single set of labels for the legend
    #plt.gcf().gca().add_artist(plt.Circle((0,0), r_fuel, fill=False, color='green', label='Fuel'))
    #plt.gcf().gca().add_artist(plt.Circle((0,0), r_clad_out, fill=False, color='red', label='Cladding'))
    #plt.gcf().gca().add_artist(plt.Rectangle((0, 0), width=0, height=0, fill=False, color='blue', label='Moderator'))
    
    # Main simulation loop for each neutron
    for i in range(num_neutrons):
        print(f"\n--- Starting Neutron {i+1} ---")
        
        alive = True
        region = 0
        
        x, y = initial_x[i], initial_y[i]
        
        cell_x_index = min(max(int((x + total_lattice_size_x/2) / pitch), 0), lattice_size_x - 1)
        cell_y_index = min(max(int((y + total_lattice_size_y/2) / pitch), 0), lattice_size_y - 1)

        
        energy = neutron_energies[i]
        
        theta = 2 * np.pi * np.random.random()
        
        path_x = [x]
        path_y = [y]
        
        while alive:
            cell_center_x = (cell_x_index - lattice_size_x/2 + 0.5) * pitch
            cell_center_y = (cell_y_index - lattice_size_y/2 + 0.5) * pitch
            x_cell = x - cell_center_x
            y_cell = y - cell_center_y
            
            # Get the pin properties for the current cell
            current_pin = lattice[cell_y_index][cell_x_index]
            has_fuel = "fuel_radius" in current_pin
            r_fuel = current_pin.get("fuel_radius", 0)
            r_clad_in = current_pin.get("clad_inner_radius", 0)
            r_clad_out = current_pin["clad_outer_radius"]
            
            sig_f, sig_c, sig_s, sig_t = get_cross_sections(energy, region)
            
            d_interaction = - (1 / sig_t) * np.log(np.random.random())
            
            d_boundary = float('inf')
            next_region = region
            
            r_sq = x_cell**2 + y_cell**2
            
            # --- Geometry Logic for the CURRENT cell ---
            if has_fuel and region == 0:  # Fuel region
                a = 1
                b = 2 * (x_cell * np.cos(theta) + y_cell * np.sin(theta))
                c = r_sq - r_clad_in**2
                delta = b**2 - 4*a*c
                if delta >= 0:
                    d_temp = (-b + np.sqrt(delta)) / (2*a)
                    if d_temp > 1e-9:
                        d_boundary = d_temp
                        next_region = 1
                        
            elif region == 1:  # Cladding region
                a = 1
                b = 2 * (x_cell * np.cos(theta) + y_cell * np.sin(theta))
                
                c_in = r_sq - r_clad_in**2
                delta_in = b**2 - 4*a*c_in
                if delta_in >= 0:
                    d_temp = (-b - np.sqrt(delta_in)) / (2*a)
                    if d_temp > 1e-9:
                        d_boundary = d_temp
                        next_region = 0

                c_out = r_sq - r_clad_out**2
                delta_out = b**2 - 4*a*c_out
                if delta_out >= 0:
                    d_temp = (-b + np.sqrt(delta_out)) / (2*a)
                    if d_temp > 1e-9 and d_temp < d_boundary:
                        d_boundary = d_temp
                        next_region = 2

            else:  # Moderator region or center of a pin1 cell
                a = 1
                b = 2 * (x_cell * np.cos(theta) + y_cell * np.sin(theta))
                
                c_out = r_sq - r_clad_out**2
                delta_out = b**2 - 4*a*c_out
                if delta_out >= 0:
                    d_temp = (-b - np.sqrt(delta_out)) / (2*a)
                    if d_temp > 1e-9:
                        d_boundary = d_temp
                        next_region = 1

                # Distance to the square cell boundaries (x_cell and y_cell are used)
                distances_to_square = []
                if np.cos(theta) > 1e-9: distances_to_square.append((pitch/2 - x_cell) / np.cos(theta))
                if np.cos(theta) < -1e-9: distances_to_square.append((-pitch/2 - x_cell) / np.cos(theta))
                if np.sin(theta) > 1e-9: distances_to_square.append((pitch/2 - y_cell) / np.sin(theta))
                if np.sin(theta) < -1e-9: distances_to_square.append((-pitch/2 - y_cell) / np.sin(theta))

                for d_sq in distances_to_square:
                    if d_sq > 1e-9 and d_sq < d_boundary:
                        d_boundary = d_sq
                        
                        temp_x = x + d_boundary * np.cos(theta)
                        temp_y = y + d_boundary * np.sin(theta)
                        
                        is_leakage = False
                        if temp_x > total_lattice_size_x/2 - 1e-9 or temp_x < -total_lattice_size_x/2 + 1e-9:
                            is_leakage = True
                        if temp_y > total_lattice_size_y/2 - 1e-9 or temp_y < -total_lattice_size_y/2 + 1e-9:
                            is_leakage = True

                        if is_leakage:
                             next_region = -1 # Leakage
                        else:
                             next_region = 3 # Special code for internal cell boundary crossing
            
            # --- Check if interaction or boundary crossing happens first ---
            if d_interaction < d_boundary:
                # print(f"  Interaction in {get_material_data(region)['name']}")
                
                x += d_interaction * np.cos(theta)
                y += d_interaction * np.sin(theta)
                path_x.append(x)
                path_y.append(y)
                
                interaction_point_x.append(x)
                interaction_point_y.append(y)
                
                rnd = np.random.random()
                if rnd <= sig_f / sig_t:
                    # print("  Event: Fission")
                    fission_count += 1
                    num_fission_neutrons = 2 if np.random.random() < 0.5 else 3
                    neutrons_produced += num_fission_neutrons
                    alive = False
                    
                elif rnd <= (sig_f + sig_c) / sig_t:
                    # print("  Event: Capture")
                    capture_count += 1
                    alive = False
                    
                else:
                    # print("  Event: Scattering")
                    scattering_count += 1
                    material_A = get_material_data(region)['A']
                    if material_A > 1:
                        ksi = 1 + np.log((material_A - 1) / (material_A + 1)) * (material_A - 1)**2 / (2 * material_A)
                    else:
                        ksi = 1
                    energy *= np.exp(-ksi * np.random.random())
                    theta = 2 * np.pi * np.random.random()
                    
            else:
                x += d_boundary * np.cos(theta)
                y += d_boundary * np.sin(theta)
                path_x.append(x)
                path_y.append(y)
                
                group = get_group(energy)
                if region == 0 and next_region == 1:
                    fuel_surf_neu_num[group] += 1
                elif region == 1 and next_region == 2:
                    clad_surf_neu_num[group] += 1
                    
                # print(f"  Boundary crossing from Region {region} to {next_region}")
                
                if next_region == -1: # Leakage from outer boundary
                    leakage_count += 1
                    
                    # Periodic boundary conditions
                    if abs(x) > total_lattice_size_x/2 - 1e-9: x = -x
                    if abs(y) > total_lattice_size_y/2 - 1e-9: y = -y
                    # print("  Event: Leakage (periodic boundary condition applied)")
                    cell_x_index = min(max(int((x + total_lattice_size_x/2) / pitch), 0), lattice_size_x - 1)
                    cell_y_index = min(max(int((y + total_lattice_size_y/2) / pitch), 0), lattice_size_y - 1)
                    region = 2
                    
                elif next_region == 3: # Internal cell boundary crossing
                    cell_x_index = min(max(int((x + total_lattice_size_x/2) / pitch), 0), lattice_size_x - 1)
                    cell_y_index = min(max(int((y + total_lattice_size_y/2) / pitch), 0), lattice_size_y - 1)
                    region = 2
                
                else:
                    region = next_region
                    
        plt.plot(path_x, path_y, '-', alpha=0.5)

    plt.plot(interaction_point_x, interaction_point_y, '*r', label='Interactions')
    plt.legend()
    plt.savefig("neutron_paths.png", dpi=300)
    plt.show()

    # --- Results and Final Plotting ---
    absorption_count = fission_count + capture_count
    total_interactions = scattering_count + absorption_count
    
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
    fuel_pins = sum(1 for row in lattice for p in row if "fuel_radius" in p)
    fuel_area = np.pi * pin["fuel_radius"]**2 * fuel_pins*8
    scaling_factor = 5 * num_neutrons
    
    flux_y = fuel_surf_neu_num / (fuel_area * scaling_factor)
    
    watt_energy = np.linspace(0.01, 30, 500)
    watt_flux = 0.453 * np.sinh(np.sqrt(2.29 * watt_energy)) * np.exp(-1.036 * watt_energy)
    
    flux_y_norm = flux_y / np.trapz(flux_y, group_energy)
    watt_flux_norm = watt_flux / np.trapz(watt_flux, watt_energy)
    
    plt.figure()
    plt.plot(group_energy, -flux_y_norm, '-bo', label='Normalized Simulated Flux')
    plt.plot(watt_energy, watt_flux_norm, '-r', label='Normalized Watt Spectrum')
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Normalized Flux")
    plt.title("Normalized Neutron Flux Spectrum")
    plt.legend(loc='upper right')
    plt.grid(True, which="both", linestyle='--')
    #plt.savefig("flux_spectrum.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    pin = {
        "fuel_radius": 0.53,
        "clad_inner_radius": 0.53,
        "clad_outer_radius": 0.90,
        "material_order": ["fuel", "clad", "moderator"]
    }
    
    pin1 = {
        "clad_outer_radius": 0.90,
        "material_order": ["clad", "moderator"]
    }
    
    # Define a 3x3 lattice with pin1 in the center
    lattice_pitch = 1.863
    lattice = [[pin, pin, pin],
               [pin, pin1, pin],
               [pin, pin, pin]]
    
    qualifying_monte_carlo(5, pin, lattice, lattice_pitch)