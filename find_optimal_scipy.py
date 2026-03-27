import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
import math
from helper_functions import calculate_overlap_area,calculate_area_outside_circle, scoring_function
from helper_functions import load_shape_from_dxf,calculate_outside_area, scale_and_rotate_dxf, mirror_and_shift_dxf
from helper_plotting import *

np.set_printoptions(precision=3, suppress=True)

"""
For a setup with up to three sensors finds the best magnetic configuration 

score used for optimization: minimum of sum of weighted angle differences
"""

# magnets used
Br = 1.345  # in Tesla
sensor_saturation = 50  # in mT
magnet_size = 0.003


# simulation parameters
num_angle_steps = 180
start_angle = 0
stop_angle = 90
ENABLE_PLOTTING = True # also at the same time enables stopping with patience

PATIENCE_LIMIT = 20       # Stop if no improvement after this many generations
MIN_IMPROVEMENT = 0.05

# solution space definition
NUM_MAGNETS = 2  # max 2

base_bounds = [
        (-0.008, -0.005),  # xm
        (-0.001, 0.008),  # ym
        (0, 360)  # rot
    ]
base_bounds_pip = [
        (-0.006, -0.00),  # xm
        (-0.002, 0.006),  # ym
        (0, 360)  # rot
    ]
force_b2b = False
base_bounds3 = [
        (-0.008, -0.002),  # xm
        (0.00, 0.008),  # ym
        (60, 120)  # rot
    ]
base_bounds2 = [
        (-0.008, -0.002),  # xm
        (0.00, 0.008),  # ym
        (240, 300)  # rot
    ]

force_front = True
base_bounds_front = [
        (-0.008, -0.0065),  # xm
        (-0.001, 0.008),  # ym
        (0, 360)  # rot
    ]
force_close_edge = False
close_dist = 0.001

clearance = 0.0008    # magnet clearance of outer ring of finger segment, so enough wall is left
clearance_magnets = 0.0003  # clearance between magnets # default 0.0005 resulting in 3.5mm magnets just touching

use_dxf_bound = True
clearance_dxf = 0.0004

# --- CHOOSE JOINT AND SET SENSOR POSITION ---
JOINT = "MCP"
number_of_sensors = 3
distance_OP = 0
r_outer = 0
xs = 0
ys = 0
xs2 = 0
ys2 = 0
xs3 = 0
ys3 = 0

if JOINT == "DIP":
    distance_OP = 0.009
    r_outer_ = 0.0045
    xs = 0.001
    ys = 0.001

elif JOINT == "PIP_base":
    distance_OP = 0.012
    r_outer = 0.006
    xs = 0.00
    ys = 0.001

elif JOINT == "MCP_v1":
    distance_OP = 0.02
    r_outer = 0.01
    xs = 0.00701
    ys = 0.00188
    xs2 = 0.00409
    ys2 = 0.00602
    # old values for designed board number 1

elif JOINT == "MCPA":  # MCP abduction
    distance_OP = 0.02
    r_outer = 0.01
    xs = 0.004
    ys = 0.00

elif JOINT == "MCP":
    distance_OP = 0.02
    r_outer = 0.01
    number_of_sensors = 3
    # --- NEW BOARDS ---
    # --- MCP ---
    # put here in mm
    base_x = 3.406
    base_y = 10.0

    xs_pcb = 2.817
    ys_pcb = 3.075

    xs2_pcb = 7.319
    ys2_pcb = 4.319

    xs3_pcb = 3.675
    ys3_pcb = 7.5

    # conversion to python program coordinates in m
    # y-> -x, x -> y
    xs = (base_y - ys_pcb) / 1000
    ys = (-base_x + xs_pcb) / 1000

    xs2 = (base_y - ys2_pcb) / 1000
    ys2 = (-base_x + xs2_pcb) / 1000

    xs3 = (base_y - ys3_pcb) / 1000
    ys3 = (-base_x + xs3_pcb) / 1000

elif JOINT == "PIP":
    number_of_sensors = 2
    distance_OP = 0.012
    r_outer = 0.006
    base_bounds = base_bounds_pip
    # --- NEW BOARDS ---
    # --- PIP ---
    # y-> -x, x -> y
    base_x = 8.266
    base_y = 6.375

    xs_pcb = 7.21
    ys_pcb = 9.222

    xs2_pcb = 10.935
    ys2_pcb = 6.622


    # conversion to python program coordinates
    xs = (-base_y + ys_pcb) / 1000
    ys = (-base_x + xs_pcb) / 1000

    xs2 = (-base_y + ys2_pcb) / 1000
    ys2 = (-base_x + xs2_pcb) / 1000

if use_dxf_bound:
    if JOINT == "MCP":
        DXF_PATH = "dxf/joint_MCP_v2.DXF"
        dxf_shape = load_shape_from_dxf(DXF_PATH)
        dxf_shape = scale_and_rotate_dxf(dxf_shape)
    elif JOINT == "PIP":
        DXF_PATH = "dxf/joint_PIP_27.DXF"
        dxf_shape = load_shape_from_dxf(DXF_PATH)
        dxf_shape = scale_and_rotate_dxf(dxf_shape, angle_deg=270)
        dxf_shape = mirror_and_shift_dxf(dxf_shape)

score_history = []
last_best_score = -np.inf
patience_counter = 0


# for repeatability
SEED = 729 # You can choose any integer
np.random.seed(SEED) # Sets the global NumPy seed

def simulate_config(xm, ym, rot, xm2, ym2, rot2, multi_output=False):
    vec_OP_base = np.array([distance_OP, 0, 0])
    vec_PM_base = np.array([xm, ym, 0])
    vec_PM_base2 = np.array([xm2, ym2, 0])

    polarisation = [Br, 0, 0]

    cube = magpy.magnet.Cuboid(polarization=polarisation,
                               dimension=(magnet_size, magnet_size, magnet_size))
    cube2 = magpy.magnet.Cuboid(polarization=polarisation,
                                dimension=(magnet_size, magnet_size, magnet_size))

    sensor = magpy.Sensor()
    sensor.position = (xs, ys, 0)

    sensor2 = magpy.Sensor()
    sensor2.position = (xs2, ys2, 0)

    sensor3 = magpy.Sensor()
    sensor3.position = (xs3, ys3, 0)

    angles_deg = np.linspace(start_angle, stop_angle, num_angle_steps)  # angle of second finger
    ori = R.from_rotvec(np.array([(0, 0, t) for t in angles_deg]), degrees=True)
    ori_half = R.from_rotvec(np.array([(0, 0, t / 2) for t in angles_deg]), degrees=True)

    ori_magnet = R.from_rotvec(np.array([(0, 0, t + rot) for t in angles_deg]), degrees=True)
    ori_magnet2 = R.from_rotvec(np.array([(0, 0, t + rot2) for t in angles_deg]), degrees=True)

    pos_P = ori_half.apply(vec_OP_base)
    vec_PM = ori.apply(vec_PM_base)
    vec_PM2 = ori.apply(vec_PM_base2)

    vec_OM = pos_P + vec_PM
    vec_OM2 = pos_P + vec_PM2

    # Set path at initialization
    cube.position = vec_OM
    cube.orientation = ori_magnet

    cube2.position = vec_OM2
    cube2.orientation = ori_magnet2

    if NUM_MAGNETS == 2:
        collection = magpy.Collection(cube, cube2)
    else:
        collection = magpy.Collection(cube)

    sensors = [sensor, sensor2, sensor3]
    sensors = sensors[0:number_of_sensors]

    B_list = [magpy.getB(collection, s) * 1000 for s in sensors]

    B = np.stack(B_list, axis=1)

    B = np.clip(B, a_min=None, a_max=sensor_saturation)

    angles = np.rad2deg(np.arctan2(B[:, :, 1], B[:, :, 0]))  # shape (num angle steps,num sen)
    magnitudes = np.linalg.norm(B[:, :, 0:2], axis=2)  # shape (num angle steps, num sen)
    if multi_output:
        score_min, score_avg, score = scoring_function(angles, magnitudes, multi_output=True)
        return score_min, score_avg, score
    else:
        score = scoring_function(angles, magnitudes)
        return score


def objective_function(x):
    # 1. Unpack variables based on mode
    if NUM_MAGNETS == 1:
        xm, ym, rot = x
        # Dummy values for the second magnet so simulate_config doesn't break
        xm2, ym2, rot2 = 0.0, 0.0, 0.0
    else:
        xm, ym, rot, xm2, ym2, rot2 = x

    if use_dxf_bound:
        if calculate_outside_area(dxf_shape,magnet_size,xm,ym,rot, clearance=clearance_dxf) > 1e-12: return np.inf
        if force_close_edge:
            if calculate_outside_area(dxf_shape, magnet_size, xm, ym, rot,
                                      clearance=close_dist) < 1e-10: return np.inf

    if calculate_area_outside_circle((xm,ym,rot), magnet_size, r_outer-clearance) > 1e-12:
        return np.inf
    if NUM_MAGNETS ==2:
        if use_dxf_bound:
            if calculate_outside_area(dxf_shape, magnet_size, xm2, ym2, rot2, clearance=clearance_dxf) > 1e-12: return np.inf
            if force_close_edge:
                if calculate_outside_area(dxf_shape, magnet_size, xm, ym, rot,
                                          clearance=close_dist) < 1e-10: return np.inf


        if calculate_area_outside_circle((xm2, ym2, rot), magnet_size, r_outer - clearance) > 1e-12:
            return np.inf
        if calculate_overlap_area((xm,ym,rot),(xm2,ym2,rot2),magnet_size+clearance_magnets) > 1e-12:
            return np.inf


    # 4. Simulation
    # We pass the dummy vars (0.0) if using 1 magnet.
    # Ensure your simulate_config handles zeros/dummy values correctly!
    score = simulate_config(xm, ym, rot, xm2, ym2, rot2)

    # Return negative because we are minimizing
    return score


def get_bounds():
    if NUM_MAGNETS == 1:
        return base_bounds
    else:
        if force_b2b:
            return base_bounds2 + base_bounds3
        elif force_front:
            return base_bounds_front + base_bounds
        else:
            return base_bounds + base_bounds


def callback_monitor(xk, convergence):
    """
    This function is called after every generation.
    xk: The best solution vector of the current generation.
    convergence: The convergence metric (fractional).

    If this returns True, optimization stops.
    """
    global last_best_score, patience_counter
    # Calculate the score of the current best vector 'xk'
    # We must negate it because objective_function returns negative score
    current_score = -objective_function(xk)
    score_history.append(current_score)

    improvement = current_score - last_best_score

    if improvement > MIN_IMPROVEMENT:
        # Significant improvement found: Reset patience
        last_best_score = current_score
        patience_counter = 0
    else:
        # No significant improvement: Count down
        patience_counter += 1

    if patience_counter >= PATIENCE_LIMIT:
        print("\n--- PATIENCE EXHAUSTED: STOPPING EARLY ---")
        return True  # Triggers the stop


def find_optimal_placement():
    print(f"Starting optimization for {NUM_MAGNETS} magnet(s)...")
    score_history.clear()
    bounds = get_bounds()


    """result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin', # best1bin or rand1bin
        maxiter=500,
        popsize=15, # default 15
        tol=0.01,  # default 0.01
        disp=True,
        polish=False,  # Disabled to prevent L-BFGS-B errors with hard constraints
        callback=callback_monitor if ENABLE_PLOTTING else None
    )"""

    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',  # best1bin nearly as good and a lot faster currenttobest1bin
        popsize=20,
        #mutation=(0.6, 1.5), # defaults seem fine
        #recombination=0.7,
        maxiter=200,
        tol=0.05,
        polish=False,
        disp=True,
        seed=SEED,
        callback=callback_monitor if ENABLE_PLOTTING else None
    )
    max_score = -result.fun
    print("\n--- Optimization Successful ---")
    print(f"Best Score: {max_score:.5f}")

    # Unpack result based on mode for printing
    if NUM_MAGNETS == 1:
        xm, ym, rot = result.x
        print(f"Magnet 1: x={xm*1000:.2f}mm, y={ym*1000:.2f}mm, rot={rot:.2f}°")
    else:
        xm, ym, rot, xm2, ym2, rot2 = result.x
        print(f"Magnet 1: x={xm*1000:.2f}mm, y={ym*1000:.2f}mm, rot={rot:.2f}°")
        print(f"Magnet 2: x={xm2*1000:.2f}mm, y={ym2*1000:.2f}mm, rot={rot2:.2f}°")

    print(", ".join(f"{val * 1000 if (i % 3) != 2 else val:.3f}" for i, val in enumerate(result.x)))

    # --- PLOTTING ---
    if ENABLE_PLOTTING:
        plt.figure(figsize=(10, 6))
        plt.plot(score_history, marker='o', linestyle='-', color='b', markersize=3)
        plt.title(f'Optimization for {JOINT} joint with {NUM_MAGNETS} magnets')
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.grid(True)
        plt.show()



    return result.x, max_score

if __name__ == "__main__":
    best_config, score = find_optimal_placement()
    if NUM_MAGNETS == 1:
        xm, ym, rot = best_config
        xm2, ym2, rot2 = 0.0, 0.0, 0.0
    else:
        xm,ym,rot,xm2,ym2,rot2 = best_config
    score_min, score_avg, score = simulate_config(xm,ym,rot,xm2,ym2,rot2, multi_output=True)
    print(f"Score min: {score_min:.2f}")
    print(f"Score avg: {score_avg:.2f}")
    print(f"Score optimized over: {score:.2f}")
    print(f"{score_min:.2f}, {score_avg:.2f}, {score:.2f}")



