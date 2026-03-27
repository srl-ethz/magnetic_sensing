import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.column import Column
from pymoo.util.display.display import Display
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pathlib import Path

import magpylib as magpy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from helper_functions import *
from helper_plotting import *

np.set_printoptions(precision=5, suppress=True)


# ---- CONFIGS ----
# magnets used
Br = 1.345  # in Tesla
sensor_saturation = 50  # in mT
magnet_size = 0.003


# simulation parameters
num_angle_steps = 180
start_angle = 0
stop_angle = 90
PLOT_SCORE = False
PLOT_PARETO = False # only for 2 objectives
PLOT_PARETO_MULTI = True

RUN_FORCED_B2B = False

PATIENCE_LIMIT = 40       # Stop if no improvement after this many generations
MIN_IMPROVEMENT = 1e-5

# solution space definition
NUM_MAGNETS = 1  # max 2
NUM_OBJECTIVES = 2

clearance_magnets = 0.0003  # clearance between magnets, (how much bigger are the simulated magnets for collision)

use_standard_clearance = False
clearance = 0.000  # magnet clearance of outer ring of finger segment, so enough wall is left

use_dxf_bound = True # used for gears
clearance_dxf = 0.0004 # actual clearance around 0.1mm lower for some reason
# dxf for pip is complete, for mcp it only covers the teeth of the gears, still need normal clearance for distance
# to cable guides, are 0.35mm deep, meaning +0.00035 higher than dxf bounds
use_dxf_bound2 = False # used for cable guide, needs to be false for PIP
clearance_dxf2 = 0.0003

name_count = "single_mag"

# [x_min, y_min, rot_min]
base_xl = [-0.008, -0.001, 0]
# [x_max, y_max, rot_max]
base_xu = [-0.003, 0.008, 360]

back_to_back_xl = np.array([-0.008, -0.001, 60, -0.008, -0.001, 240])
back_to_back_xu = np.array([-0.003, 0.008, 120,  -0.003, 0.008, 300])


base_xl_pip = [-0.006, -0.002, 0]
base_xu_pip = [0.00, 0.006, 360]

back_to_back_xl_pip = np.array([-0.006, -0.002, 60, -0.006, -0.002, 240])
back_to_back_xu_pip = np.array([0.0, 0.006, 120,  0.0, 0.006, 300])


# --- CHOOSE JOINT AND SET SENSOR POSITION ---
JOINT = "PIP"
number_of_sensors = 3
distance_OP = 0
r_outer = 0
xs = 0
ys = 0
xs2 = 0
ys2 = 0
xs3 = 0
ys3 = 0

if JOINT == "MCP":
    distance_OP = 0.02 #should be 2x radius of rolling surface
    r_outer = 0.01 # radius of rolling surface
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
    base_xl = base_xl_pip
    base_xu = base_xu_pip
    back_to_back_xl = back_to_back_xl_pip
    back_to_back_xu = back_to_back_xu_pip
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



SEED = 729 # You can choose any integer
np.random.seed(SEED) # Sets the global NumPy seed

if use_dxf_bound:
    if JOINT == "MCP":
        DXF_PATH = "dxf/joint_MCP_gears.DXF"
        dxf_shape = load_shape_from_dxf(DXF_PATH)
        dxf_shape = scale_and_rotate_dxf(dxf_shape)
    elif JOINT == "PIP":
        DXF_PATH = "dxf/joint_PIP_27.DXF"
        dxf_shape = load_shape_from_dxf(DXF_PATH)
        dxf_shape = scale_and_rotate_dxf(dxf_shape, angle_deg=270)
        dxf_shape = mirror_and_shift_dxf(dxf_shape)
if use_dxf_bound2:
    if JOINT == "MCP":
        DXF_PATH2 = "dxf/joint_MCP_cable_guide.DXF"
        dxf_shape2 = load_shape_from_dxf(DXF_PATH2)
        dxf_shape2 = scale_and_rotate_dxf(dxf_shape2)

# ==========================================
# SIMULATION
# ==========================================
def simulate_config(xm, ym, rot, xm2, ym2, rot2):
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
    score_min, score_avg, score = scoring_function(angles, magnitudes, multi_output=True)
    return score_min, score_avg




# ==========================================
# 2. The Pymoo Problem Class
# ==========================================


class MagnetPlacementProblem(ElementwiseProblem):

    def __init__(self, xl=None, xu=None):
        if xl is None:
            xl_used = np.tile(base_xl, NUM_MAGNETS)
        else:
            xl_used = xl
        if xu is None:
            xu_used = np.tile(base_xu, NUM_MAGNETS)
        else:
            xu_used = xu
        total_vars = 3 * NUM_MAGNETS


        self.n_collision_checks = 1 if NUM_MAGNETS > 1 else 0

        total_constraints = self.n_collision_checks
        if use_standard_clearance:
            total_constraints += NUM_MAGNETS
        if use_dxf_bound:
            total_constraints += NUM_MAGNETS
        if use_dxf_bound2:
            total_constraints+= NUM_MAGNETS



        super().__init__(
            n_var = total_vars,
            n_obj=NUM_OBJECTIVES,  # 1 Objective (Score)
            n_ieq_constr=total_constraints,
            xl=xl_used,  # Lower Bounds
            xu=xu_used
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Unpack variables
        if NUM_MAGNETS == 2:
            xm, ym, rot, xm2, ym2, rot2 = x
        else:
            xm, ym, rot = x
            xm2,ym2,rot2 = 0,0,0

        # 2. Calculate Objective (Score)
        # Note: Pymoo always MINIMIZES. So we return -score.
        score_min, score_avg = simulate_config(xm, ym, rot, xm2, ym2, rot2)
        if NUM_OBJECTIVES == 2:
            out["F"] = [-score_min,-score_avg]
        else:
            out["F"] = [-(score_min+0.2*score_avg)]

        # 3. Calculate Constraints (Collision)
        # Pymoo expects: g(x) <= 0 is VALID. g(x) > 0 is INVALID.



        g = []
        if use_standard_clearance:
            g.append(calculate_area_outside_circle((xm,ym,rot), magnet_size, r_outer-clearance))
        if use_dxf_bound:
            g.append(calculate_outside_area(dxf_shape,magnet_size,xm,ym,rot,clearance=clearance_dxf))
        if use_dxf_bound2:
            g.append(calculate_outside_area(dxf_shape2,magnet_size,xm,ym,rot,clearance=clearance_dxf2))

        if NUM_MAGNETS == 2:
            g.append(calculate_overlap_area((xm, ym, rot), (xm2, ym2, rot2), magnet_size + clearance_magnets))

            if use_standard_clearance:
                g.append(calculate_area_outside_circle((xm2, ym2, rot2), magnet_size, r_outer - clearance))
            if use_dxf_bound:
                g.append(calculate_outside_area(dxf_shape, magnet_size, xm2, ym2, rot2, clearance=clearance_dxf))
            if use_dxf_bound2:
                g.append(calculate_outside_area(dxf_shape2, magnet_size, xm2, ym2, rot2, clearance=clearance_dxf2))

        out["G"] = np.array(g)


# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":
    problem = MagnetPlacementProblem()
    problem_b2b = MagnetPlacementProblem(xl=back_to_back_xl, xu=back_to_back_xu)

    # 2. Configure the Algorithm (NSGA-II)
    algorithm = NSGA2(
        pop_size=400,  # 1. Double the scouts
        n_offsprings=400,  # 2. Force 100% turnover
        sampling=LHS(),  # 3. Mathematically perfect even spread

        # CROSSOVER (Mixing Parents)
        # eta=10: Lower value means children are created MUCH further away from parents
        crossover=SBX(prob=0.9, eta=11),

        # MUTATION (Random Kicks)
        # prob=0.3: Force 30% of variables to mutate (Default is usually ~16%)
        # eta=10:   When they do mutate, kick them hard across the bounds
        mutation=PM(prob=0.3, eta=11),

        eliminate_duplicates=True  # Essential to prevent cloning
    )

    # 3. Define Termination
    termination = DefaultSingleObjectiveTermination(
        xtol=1e-6,       # Stop if position changes < 0.000001
        ftol=0.05,       # Stop if score changes < 0.1
        period=20,       # Check for convergence every 20 generations (replaces 'n_last')
        n_max_gen=400    # Hard limit
    )

    print("Starting Pymoo Optimization for Standard")

    # 4. Run!
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=SEED,
        save_history=True,
        verbose=True
    )
    dir_name = f"sim_results/{JOINT}/0{(clearance * 1e4):.0f}_dxf_0{(clearance_dxf * 1e4):.0f}_dxf2_0{(clearance_dxf2 * 1e4):.0f}"
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    save_results_to_csv(res, f"{dir_name}/standard_{name_count}.csv")

    if RUN_FORCED_B2B and NUM_MAGNETS == 2:
        print("Starting Pymoo Optimization for forced B2B solution")
        res_b2b = minimize(
            problem_b2b,
            algorithm,
            termination,
            seed=SEED+1,
            save_history=True,
            verbose=True
        )
        save_results_to_csv(res_b2b,f"{dir_name}/b2b_{name_count}.csv")
        if PLOT_PARETO_MULTI:
            plot_combined_pareto(res, res_b2b)
        # --- MERGE LOGIC ---
        # Check if both runs actually found valid solutions
        if res.X is not None and res_b2b.X is not None:

            # 1. Ensure arrays are 2D (Pymoo sometimes returns 1D if only 1 solution is found)
            X1, F1 = np.atleast_2d(res.X), np.atleast_2d(res.F)
            X2, F2 = np.atleast_2d(res_b2b.X), np.atleast_2d(res_b2b.F)

            # 2. Stack them together into a master pool
            combined_X = np.vstack((X1, X2))
            combined_F = np.vstack((F1, F2))

            # 3. Filter the master pool to find the true Global Pareto Front
            # only_non_dominated_front=True returns just the indices of the ultimate winners
            front_indices = NonDominatedSorting().do(combined_F, only_non_dominated_front=True)

            # 4. Overwrite 'res' so the rest of your script works natively
            res.X = combined_X[front_indices]
            res.F = combined_F[front_indices]

        elif res_b2b.X is not None:
            # If the first run completely failed but B2B succeeded, just copy B2B over
            res.X = res_b2b.X
            res.F = res_b2b.F


    print("\n--- Optimization Finished ---")
    if res.X is not None:
        if NUM_OBJECTIVES == 1:
            print(f"Best Score: {-res.F[0]}")  # Invert back to positive
            if NUM_MAGNETS == 1:
                xm, ym, rot = res.X
                print(f"Magnet 1: x={xm * 1000:.2f}mm, y={ym * 1000:.2f}mm, rot={rot:.2f}°")
            else:
                xm, ym, rot, xm2, ym2, rot2 = res.X
                print(f"Magnet 1: x={xm * 1000:.2f}mm, y={ym * 1000:.2f}mm, rot={rot:.2f}°")
                print(f"Magnet 2: x={xm2 * 1000:.2f}mm, y={ym2 * 1000:.2f}mm, rot={rot2:.2f}°")

            print(", ".join(f"{val * 1000 if (i % 3) != 2 else val:.3f}" for i, val in enumerate(res.X)))
        else:
            # --- PRINTING OUTPUT ---
            print(f"Found {len(res.X)} optimal solutions on the Pareto Front:\n")

            # Pair the solutions and scores, then sort them
            # item[1] is the score array 'f'. item[1][0] is f[0].
            # We sort by -f[0] (which is Obj1) in descending order (reverse=True) so the highest score is first.
            sorted_solutions = sorted(zip(res.X, res.F), key=lambda item: -item[1][0], reverse=True)

            # Loop through the sorted solutions
            for index, (x, f) in enumerate(sorted_solutions):
                # 1. Format the variables (x,y in mm, rot in degrees)
                var_str = ", ".join(f"{val * 1000 if (i % 3) != 2 else val:.6f}" for i, val in enumerate(x))

                # 2. Format the scores (flip back to positive)
                score_1 = -f[0]
                score_2 = -f[1]

                # 3. Print the combined output
                print(f"Solution {index + 1}:")
                print(f"  Scores : Min, Avg, Min+0.3Avg = {score_1:.3f}, {score_2:.3f}, {(score_1+0.3*score_2):.2f}")
                print(f"  Vars   : {var_str}")
                print("-" * 40)

            # --- PLOT PARETO FRONT ---
            if PLOT_PARETO:
                f1_values = -res.F[:, 0]
                f2_values = -res.F[:, 1]

                # Create the plot
                plt.figure(figsize=(8, 6))
                plt.scatter(f1_values, f2_values, color='dodgerblue', edgecolor='k', alpha=0.8, label='Optimal Trade-offs')

                # Formatting the chart
                plt.title("Pareto Front: Score min vs Score avg")
                plt.xlabel("Score min")
                plt.ylabel("Score avg")
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.legend()

                # Show the plot
                plt.show()

    else:
        print("No valid solution found.")
    if res.history:
        if PLOT_SCORE:
            plot_score_evolution(res)