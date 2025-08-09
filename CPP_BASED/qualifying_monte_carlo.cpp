#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <map>

// Using a consistent random number generator
std::mt19937_64 rng(0);
std::uniform_real_distribution<double> dist(0.0, 1.0);

// Helper function to get material properties
std::map<std::string, double> get_material_data(int region) {
    if (region == 0) { // Fuel
        return {{"name", 0.0}, {"A", 238.02891}, {"density", 10.97}};
    } else if (region == 1) { // Cladding
        return {{"name", 1.0}, {"A", 26.981539}, {"density", 2.70}};
    } else { // Moderator
        return {{"name", 2.0}, {"A", 1.00794}, {"density", 1.0}};
    }
}

// Helper function to get the energy group
int get_group(double energy) {
    std::vector<double> group_energy_boundaries = {3e+1, 3e+0, 3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7, 3e-8};
    int group = 9;
    for (int g = 0; g < group_energy_boundaries.size(); ++g) {
        if (energy >= group_energy_boundaries[g]) {
            group = g;
            break;
        }
    }
    return group;
}

// Function to get macroscopic cross sections
std::vector<double> get_cross_sections(double energy, int region) {
    std::vector<std::vector<double>> sigma_f = {
        {1.05e-1, 0, 0}, {5.96e-2, 0, 0}, {6.02e-2, 0, 0}, {1.06e-1, 0, 0},
        {2.46e-1, 0, 0}, {2.50e-1, 0, 0}, {1.07e-1, 0, 0}, {1.28e+0, 0, 0},
        {9.30e+0, 0, 0}, {2.58e+1, 0, 0}
    };
    std::vector<std::vector<double>> sigma_c = {
        {1.41e-6, 1.71e-2, 3.34e-6}, {1.34e-3, 7.83e-3, 3.34e-6}, {1.10e-2, 2.83e-4, 2.56e-7},
        {3.29e-2, 4.52e-6, 6.63e-7}, {8.23e-2, 1.06e-5, 2.24e-7}, {4.28e-2, 4.39e-6, 1.27e-7},
        {9.90e-2, 1.25e-5, 2.02e-7}, {2.51e-1, 3.98e-5, 6.02e-7}, {2.12e+0, 1.26e-4, 1.84e-6},
        {4.30e+0, 3.95e-4, 5.76e-6}
    };
    std::vector<std::vector<double>> sigma_s = {
        {2.76e-1, 1.44e-1, 1.27e-2}, {3.88e-1, 1.76e-1, 7.36e-2}, {4.77e-1, 3.44e-1, 2.65e-1},
        {6.88e-1, 2.66e-1, 5.72e-1}, {9.38e-1, 2.06e-1, 6.69e-1}, {1.52e+0, 2.14e-1, 6.81e-1},
        {2.30e+0, 2.23e-1, 6.82e-1}, {2.45e+0, 2.31e-1, 6.83e-1}, {9.79e+0, 2.40e-1, 6.86e-1},
        {4.36e+1, 2.41e-1, 6.91e-1}
    };

    int group = get_group(energy);

    double sig_f = sigma_f[group][region];
    double sig_c = sigma_c[group][region];
    double sig_s = sigma_s[group][region];
    double sig_t = sig_f + sig_c + sig_s;

    return {sig_f, sig_c, sig_s, sig_t};
}

void qualifying_monte_carlo(int num_neutrons) {
    std::cout << "Number of Neutrons.......................= " << num_neutrons << std::endl;

    // --- Geometry Parameters ---
    const double r_fuel = 0.53;
    const double r_clad_in = 0.53;
    const double r_clad_out = 0.90;
    const double pitch = 1.837;

    // --- Simulation Counters ---
    long long fission_count = 0;
    long long capture_count = 0;
    long long scattering_count = 0;
    long long leakage_count = 0;
    long long neutrons_produced = 0;

    // --- Neutron Initial State Generation ---
    std::vector<double> neutron_energies;
    std::exponential_distribution<double> maxwellian_dist(1.0); // Simple Maxwellian approx
    for (int i = 0; i < num_neutrons; ++i) {
        neutron_energies.push_back(maxwellian_dist(rng));
    }

    std::vector<double> initial_x(num_neutrons);
    std::vector<double> initial_y(num_neutrons);
    for (int i = 0; i < num_neutrons; ++i) {
        double r_rand = dist(rng) * dist(rng); // Simulating a power law for radius
        double radius = r_rand * r_fuel;
        double theta_initial = 2.0 * M_PI * dist(rng);
        initial_x[i] = radius * std::cos(theta_initial);
        initial_y[i] = radius * std::sin(theta_initial);
    }

    // Main simulation loop for each neutron
    for (int i = 0; i < num_neutrons; ++i) {
        std::cout << "\n--- Starting Neutron " << i + 1 << " ---" << std::endl;

        bool alive = true;
        int region = 0;
        double x = initial_x[i];
        double y = initial_y[i];
        double energy = neutron_energies[i];
        double theta = 2.0 * M_PI * dist(rng);

        while (alive) {
            std::vector<double> cross_sections = get_cross_sections(energy, region);
            double sig_f = cross_sections[0];
            double sig_c = cross_sections[1];
            double sig_s = cross_sections[2];
            double sig_t = cross_sections[3];

            if (sig_t == 0) {
                // Prevent division by zero, neutron gets lost
                break;
            }

            double d_interaction = - (1.0 / sig_t) * std::log(dist(rng));
            double d_boundary = std::numeric_limits<double>::infinity();
            int next_region = region;

            double r_sq = x * x + y * y;

            if (region == 0) { // Fuel
                double b = 2.0 * (x * std::cos(theta) + y * std::sin(theta));
                double c = r_sq - r_clad_in * r_clad_in;
                double delta = b * b - 4.0 * c;
                if (delta >= 0) {
                    double d_temp = (-b + std::sqrt(delta)) / 2.0;
                    if (d_temp > 1e-9) {
                        d_boundary = d_temp;
                        next_region = 1;
                    }
                }
            } else if (region == 1) { // Cladding
                double b = 2.0 * (x * std::cos(theta) + y * std::sin(theta));
                double c_in = r_sq - r_clad_in * r_clad_in;
                double delta_in = b * b - 4.0 * c_in;
                if (delta_in >= 0) {
                    double d_temp = (-b - std::sqrt(delta_in)) / 2.0;
                    if (d_temp > 1e-9) {
                        d_boundary = d_temp;
                        next_region = 0;
                    }
                }
                double c_out = r_sq - r_clad_out * r_clad_out;
                double delta_out = b * b - 4.0 * c_out;
                if (delta_out >= 0) {
                    double d_temp = (-b + std::sqrt(delta_out)) / 2.0;
                    if (d_temp > 1e-9 && d_temp < d_boundary) {
                        d_boundary = d_temp;
                        next_region = 2;
                    }
                }
            } else if (region == 2) { // Moderator
                double b = 2.0 * (x * std::cos(theta) + y * std::sin(theta));
                double c_out = r_sq - r_clad_out * r_clad_out;
                double delta_out = b * b - 4.0 * c_out;
                if (delta_out >= 0) {
                    double d_temp = (-b - std::sqrt(delta_out)) / 2.0;
                    if (d_temp > 1e-9) {
                        d_boundary = d_temp;
                        next_region = 1;
                    }
                }

                if (std::abs(std::cos(theta)) > 1e-9) {
                    double d_sq = (pitch/2.0 - x) / std::cos(theta);
                    if (d_sq > 1e-9 && d_sq < d_boundary) {
                        d_boundary = d_sq;
                        next_region = -1;
                    }
                }
                if (std::abs(std::cos(theta)) > 1e-9) {
                    double d_sq = (-pitch/2.0 - x) / std::cos(theta);
                    if (d_sq > 1e-9 && d_sq < d_boundary) {
                        d_boundary = d_sq;
                        next_region = -1;
                    }
                }
                if (std::abs(std::sin(theta)) > 1e-9) {
                    double d_sq = (pitch/2.0 - y) / std::sin(theta);
                    if (d_sq > 1e-9 && d_sq < d_boundary) {
                        d_boundary = d_sq;
                        next_region = -1;
                    }
                }
                if (std::abs(std::sin(theta)) > 1e-9) {
                    double d_sq = (-pitch/2.0 - y) / std::sin(theta);
                    if (d_sq > 1e-9 && d_sq < d_boundary) {
                        d_boundary = d_sq;
                        next_region = -1;
                    }
                }
            }

            if (d_interaction < d_boundary) {
                // Interaction
                x += d_interaction * std::cos(theta);
                y += d_interaction * std::sin(theta);

                double rnd = dist(rng);
                if (rnd <= sig_f / sig_t) {
                    fission_count++;
                    neutrons_produced += (dist(rng) < 0.5) ? 2 : 3;
                    alive = false;
                } else if (rnd <= (sig_f + sig_c) / sig_t) {
                    capture_count++;
                    alive = false;
                } else {
                    scattering_count++;
                    double material_A = get_material_data(region)["A"];
                    double ksi = 1.0 + std::log((material_A - 1.0) / (material_A + 1.0)) * (material_A - 1.0) * (material_A - 1.0) / (2.0 * material_A);
                    energy *= std::exp(-ksi * dist(rng));
                    theta = 2.0 * M_PI * dist(rng);
                }
            } else {
                // Boundary crossing
                x += d_boundary * std::cos(theta);
                y += d_boundary * std::sin(theta);

                if (next_region == -1) {
                    leakage_count++;
                    if (std::abs(x) > pitch / 2.0 - 1e-9) { x = -x; }
                    if (std::abs(y) > pitch / 2.0 - 1e-9) { y = -y; }
                    region = 2; // Re-enter moderator
                } else {
                    region = next_region;
                }
            }
        }
    }

    // --- Results ---
    long long absorption_count = fission_count + capture_count;
    long long total_interactions = scattering_count + absorption_count;
    double keff = 0.0;
    if (absorption_count + leakage_count > 0) {
        keff = static_cast<double>(neutrons_produced + leakage_count) / (absorption_count + leakage_count);
    }

    std::cout << "\n--- Simulation Results ---" << std::endl;
    std::cout << "Number of Neutrons.......................= " << num_neutrons << std::endl;
    std::cout << "Total Interactions.......................= " << total_interactions << std::endl;
    std::cout << "  Scattering Events......................= " << scattering_count << std::endl;
    std::cout << "  Capture Events.........................= " << capture_count << std::endl;
    std::cout << "  Fission Events.........................= " << fission_count << std::endl;
    std::cout << "  Total Absorption Events................= " << absorption_count << std::endl;
    std::cout << "  Leakage Events.........................= " << leakage_count << std::endl;
    std::cout << "Neutrons Produced by Fission.............= " << neutrons_produced << std::endl;
    std::cout << "Effective Multiplication Factor (keff)...= " << keff << std::endl;
}

int main() {
    qualifying_monte_carlo(10); // Example run
    return 0;
}
