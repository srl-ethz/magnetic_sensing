import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
from helper_functions import *
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


# solution space definition
NUM_MAGNETS = 2  # max 2
# --- ENTER MAGNET POSITONS HERE [mm,°]


xm_base, ym_base, base_rot_mag, xm_base2, ym_base2, base_rot_mag2 = -4.217, 6.142, 109.416, -7.353, 0.504, 264.211
xm_base, ym_base, base_rot_mag, xm_base2, ym_base2, base_rot_mag2 = -4.217, 6.142, 0, -7.353, 0.504, 90
#xm_base, ym_base, base_rot_mag, xm_base2, ym_base2, base_rot_mag2 = -7.299, -0.265, 348.933, -4.660, 5.713, 329.195

#xm_base, ym_base, base_rot_mag, xm_base2, ym_base2, base_rot_mag2 = -0.981, 3.071, 129.713, -3.008, -0.894, 181.596

#xm_base, ym_base, base_rot_mag, xm_base2, ym_base2, base_rot_mag2 = -6.774, 1.764, 266.844, 0,0,0





SHOW = True
PLOT_BASE = False
as_polar = False
PLOT_INFO = False

if 0:
    base_rot_mag-=180
    base_rot_mag2-=180


# change to SI unit (meter)
xm_base *= 1e-3
ym_base *= 1e-3
xm_base2 *= 1e-3
ym_base2 *= 1e-3


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
    if SHOW:
        if NUM_MAGNETS == 2:
            magpy.show(cube, cube2, sensor, sensors, animation=True, backend="plotly")
        else:
            magpy.show(cube, sensor, sensors, animation=True, backend="plotly")


    B_list = [magpy.getB(collection, s) * 1000 for s in sensors]

    B = np.stack(B_list, axis=1)

    B = np.clip(B, a_min=None, a_max=sensor_saturation)


    angles = np.rad2deg(np.arctan2(B[:, :, 1], B[:, :, 0]))  # shape (num angle steps,num sen)
    magnitudes = np.linalg.norm(B[:, :, 0:2], axis=2)  # shape (num angle steps, num sen)
    if PLOT_BASE:
        plot_magnetic_field(angles,magnitudes, as_polar=as_polar, show_range=True, x_range=(start_angle,stop_angle))
    if PLOT_INFO:
        plot_sensor_information(angles,magnitudes,x_range=(start_angle,stop_angle))
    score = scoring_function(angles, magnitudes, multi_output=True, joint_range=(stop_angle-start_angle))
    return score


if __name__ == "__main__":
    score = simulate_config(xm_base,ym_base,base_rot_mag,xm_base2,ym_base2,base_rot_mag2)
    print(f"Score min: {score[0]}")
    print(f"Score avg: {score[1]}")
    print(f"Score trained: {score[2]}")


