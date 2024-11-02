from ground_station_simulations import satellite_path as sp
from ground_station_simulations import plotting as pl

def get_best_solution(orbits_params, sequence_type="All", plotting=False):

    if plotting:
        ax = pl.new_figure()
    else:
        ax = None

    if sequence_type == "hohmann-like":
        print("\n\n---Running hohmann strategy without phasing---------------------------------\n")

    if sequence_type == "hohmann-like-with-phasing":
        print("\n\n---Running hohmann strategy with phasing------------------------------------\n")


    # Get best permutation of orbits of given sequence type
    path = get_best_permutation(orbits_params, sequence_type, plotting, ax)
    
    # Print delta-vs (scalars) with labels and timestamps
    print("  Delta-v Scalars (magnitude of delta-v for each maneuver):")
    for i, dv in enumerate(path.dvs):
        maneuver_time = path.time_array_segments[i+1][0]
        print(f"    Maneuver {i + 1} at t = {maneuver_time:.3f} s: {dv:.3f} km/s")

    # Print delta-v vectors with labels and timestamps
    print("\n  Delta-v Vectors (direction and magnitude of delta-v for each maneuver):")
    for i, dv_vec in enumerate(path.dv_vecs):
        maneuver_time = path.time_array_segments[i+1][0]
        print(f"    Maneuver {i + 1} at t = {maneuver_time:.3f} s: [{dv_vec[0]:.5f}, {dv_vec[1]:.5f}, {dv_vec[2]:.5f}] km/s")
        
    if plotting:
        for i, orbit_params in enumerate(orbits_params):
            target_path = plot_target(orbit_params)

            pl.plot_path(ax, target_path, color_offset=i/len(orbits_params), Earth=False, base_label=f"Target: {i}", label_ends=False, linestyle='--')

        pl.plot_path(ax, path)

        # Show anything that was plotted
        ax.legend()
        pl.show(ax)

        # pl.animate_path(path, color_offset=0, base_label="Project Icarus", Earth=True, speed_factor=30)

    return path

def get_best_permutation(orbits_params, sequence_type, plotting=False, ax=None):
    transfer_dvs = {}

    # Find the best path for each transfer maneuver
    for i in orbits_params:
        for j in orbits_params:
            if i == j:
                continue
            else: 
                path = sp.SatellitePath()
                path.generate_path((i, j), sequence_type=sequence_type, plotting=False)
                transfer_dvs[f'{i} + {j}'] = path.dv_total

    # Generate all permutations of the four orbits (indices 1 to 4)
    manoeuvre_combos = permute_orbits([0, 1, 2, 3])
    print("\nSearching permutations of target orbits...\n")
    
    total_delta_vs = []
    
    for idx, combo in enumerate(manoeuvre_combos, 1):
        formatted_combo = " -> ".join(chr(65 + i) for i in combo)  # Convert indices to letters (A, B, C, D)
        # Calculate total delta-v for the current permutation
        total_delta_v = sum(
            transfer_dvs[f'{orbits_params[combo[i]]} + {orbits_params[combo[i+1]]}']
            for i in range(len(combo) - 1)
        )
        total_delta_vs.append(total_delta_v)
        
        print(f"  Permutation {idx}: {formatted_combo} | Total Delta-v: {total_delta_v:.3f} km/s")

    # Find the best maneuver (minimum delta_v)
    best_manoeuvre_index = total_delta_vs.index(min(total_delta_vs))
    best_combo = manoeuvre_combos[best_manoeuvre_index]
    best_delta_v = min(total_delta_vs)

    # Reorder the satellite array to match the best combo
    reordered_satellite_array = [orbits_params[i] for i in best_combo]
    
    # Print the optimal order of indices with delta-v
    formatted_best_combo = " -> ".join(chr(65 + i) for i in best_combo)
    print(f"\nBest transfer permutation found! : {formatted_best_combo}")
    print(f"  Delta-v total: {best_delta_v:.3f} km/s")

    best_path = sp.SatellitePath()

    best_path.generate_path(reordered_satellite_array,
                            sequence_type=sequence_type,
                            plotting=plotting, ax=ax)
    return best_path


def permute_orbits(arr):
    if len(arr) == 1:
        return [arr]
    
    parking_orbit = arr[0]
    remaining = arr[1:]
    
    result = []
    for i in range(len(remaining)):
        current = remaining[i]
        rest = remaining[:i] + remaining[i+1:]
        for p in permute_orbits([current] + rest):
            result.append([parking_orbit] + p)
    
    return result

def plot_target(params):
    path = sp.SatellitePath()
    path.generate_path((params,), "coast", plotting=True)

    return path

