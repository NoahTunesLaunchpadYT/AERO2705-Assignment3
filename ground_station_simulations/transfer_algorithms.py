import plotting as pl
import satellite_path as sp
import linear_algebra as la
import definitions as d
import orbit as o

def get_best_solution(orbits, sequence_type="All", plotting=True):
    
    if plotting:
        # Start plotting
        ax = pl.new_figure()

    # Get best permutation of orbits of given sequence type
    path = get_best_permutation(orbits, sequence_type, plotting)

    if plotting:
        # Show anything that was plotted
        pl.plot_path(ax, path)
        pl.show(ax)

    return path

def get_best_permutation(orbits_params, sequence_type, plotting):
    transfer_dvs = {}

    # Find the best path for each transfer manoeuvre
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
    print("\nPERMUTATIONS")
    print(manoeuvre_combos)
    print("\n")
        
    total_delta_vs = []
    
    for combo in manoeuvre_combos:
        total_delta_v = sum(
            transfer_dvs[f'{orbits_params[combo[i]]} + {orbits_params[combo[i+1]]}']
            for i in range(len(combo) - 1)
        )
        total_delta_vs.append(total_delta_v)
    
    # Find the best manoeuvre (minimum delta_v)
    best_manoeuvre_index = total_delta_vs.index(min(total_delta_vs))
    best_combo = manoeuvre_combos[best_manoeuvre_index]

    # Reorder the satellite array to match the best combo
    reordered_satellite_array = [orbits_params[i] for i in best_combo]
    
    print("Best Delta v:", min(total_delta_vs))
    print("Optimal Order of Indices:", best_combo)

    best_path = sp.SatellitePath()

    best_path.generate_path(reordered_satellite_array,
                            sequence_type=sequence_type,
                            plotting=plotting)



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