def plot_geometry():
    r_fuel = 0.53
    r_clad_out = 0.90
    pitch = 1.837

    fig, ax = plt.subplots(figsize=(8, 8))
    # Fuel Geometry 
    fuel_circle = plt.Circle((0, 0), r_fuel, fill=False, color='green', linewidth=2, label='Fuel')
    ax.add_artist(fuel_circle)

    # Cladding Geometry
    clad_circle = plt.Circle((0, 0), r_clad_out, fill=False, color='red', linewidth=2, label='Cladding')
    ax.add_artist(clad_circle)

    # Pitch
    mod_square = plt.Rectangle((-pitch/2, -pitch/2), pitch, pitch, fill=False, color='blue', linewidth=2, label='Moderator')
    ax.add_artist(mod_square)

    ax.set_xlim(-pitch/2 - 0.1, pitch/2 + 0.1)
    ax.set_ylim(-pitch/2 - 0.1, pitch/2 + 0.1)
    ax.set_aspect('equal')
    ax.set_title("Fuel Cell Geometry")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.legend()
    ax.grid(True, linestyle='--')
    plt.savefig("geometry.png", dpi=300)
    plt.show()
plot_geometry()    