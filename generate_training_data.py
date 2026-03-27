import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
import pandas as pd


np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.3f}'.format})


"""this is a modelling of a sensor and a magnet on the pip joint
    the origin is placed on center of the first phalanx (point O)
    the center of the second phalanx is point P
    the offset of the sensor to O, and the magnet to point P resp., is set
"""

# ----- CONFIG -----
filename = 'training_data/sim_pcb2_exp3_2_1.csv'  # save data to this

#CSV_LOAD_PATH = 'C:/Users/laure/PycharmProjects/MT_joint_angle_sensing/classic_data/pcb2_mcp_5.csv'
CSV_LOAD_PATH = 'C:/Users/laure/PycharmProjects/MT_joint_angle_sensing/logs/PCB_v2/Experiment 3/mcp_2.csv'
load_from_csv = True
# ------ TRAINING PARAMS -----
num_positions = 2000  # training positions per angle
size_training_mag = 0.003
min_training_radius = 0.007  # min distance of center of magnet to center of two sensors [m]
max_training_radius = 0.020  # max distance of center of magnet to center of two sensors [m]

reps_of_baseline = 10 # how many times is the position with no training magnet added

add_noise = False
noise_factor = 0.01

double_outliers= True

keepout_on_finger = False
# for MCP
x_min, x_max = -36.0*1e-3,  4.5*1e-3  # in m
y_min, y_max = -7.0*1e-3, 5.0*1e-3
z_min, z_max = -1.0*1e-3,  10.0*1e-3


# size in meters
magnet_size_x = 0.003  # 0.002
magnet_size_y = 0.003  # 0.005
magnet_size_z = 0.003  # 0.0025

magnet_size_z = magnet_size_x
magnet_size_y = magnet_size_x
Br = 1.18 #1.345 from manufacturer, measured only 1.18 for 3mm magnet # in Tesla
offset_z = -0.0013

num_angle_steps = 90+1
start_angle = 0
stop_angle = 90

#  all in meters
distance_OP = 0.02
JOINT = "MCP"
double_sensor = True
triple_sensor = True
sensor1_rot = 0
sensor2_rot = 0
sensor3_rot = 0

if JOINT == "MCP":
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

    sensor1_rot = 0
    sensor2_rot = 125
    sensor3_rot = 0

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

    sensor1_rot = 0
    sensor2_rot = 0
    triple_sensor = False
    xs3=1e5
    ys3=1e5
    sensor3_rot = 0
else:
    print("Error case not defined")
    exit(0)




xm,ym,base_rot_mag = -7.347,   0.712, 175.


show = False

flip_sensor = False
angle_shift = 0 # angle shift in deg, to find matching
shift_sensors_x = 0 # shift sensors in mm
shift_sensors_y = 0 # shift sensors in mm

second_magnet = True
xm2,ym2,base_rot_mag2 = -4.983,   4.983, 150.

if 1:
    base_rot_mag+=180
    base_rot_mag2+=180
xs += shift_sensors_x/1000
xs2 += shift_sensors_x/1000
xs3 += shift_sensors_x/1000
ys += shift_sensors_y/1000
ys2 += shift_sensors_y/1000
ys3 += shift_sensors_y/1000


# change to SI unit (meter)
xm *= 1e-3
ym *= 1e-3
xm2 *= 1e-3
ym2 *= 1e-3

# ----- HELPER FUNTIONS -----
def get_n_points_in_shell(n, dist_min, dist_max):
    """
    Returns an (n, 3) numpy array of n random points uniformly distributed
    within the spherical shell defined by dist_min and dist_max.
    """
    # 1. Generate n random directions
    # Shape is (n, 3) -> n rows, 3 columns (x, y, z)
    points = np.random.normal(0, 1, size=(n, 3))

    # 2. Normalize vectors to be unit length
    # axis=1 means we calculate the norm across the row (x,y,z)
    # keepdims=True ensures the shape remains (n, 1) for broadcasting
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit_vectors = points / norms

    # 3. Generate n random radii
    # We generate n random numbers 'u' between 0 and 1
    u = np.random.random(size=(n, 1))

    # Apply the cube root formula for volumetric uniformity
    radii = (u * (dist_max ** 3 - dist_min ** 3) + dist_min ** 3) ** (1 / 3)

    # 4. Scale the directions by the radii
    return unit_vectors * radii

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    expected_cols = ['x_1','y_1','z_1','x_2','y_2','z_2','angle_joint']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"CSV must contain columns: {expected_cols}. Found: {list(df.columns)}")
    if triple_sensor:
        X = df[['x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3']].values
    else:
        X = df[['x_1','y_1','z_1','x_2','y_2','z_2']].values
    y = df['angle_joint'].values
    return X, y


# ----- CODE -----
sensor = magpy.Sensor()
sensor.position = (xs, ys, 0)
sensor.orientation = R.from_rotvec(np.array([0, 0, sensor1_rot]), degrees=True)

sensor2 = magpy.Sensor()
sensor2.position = (xs2, ys2, 0)
sensor2.orientation = R.from_rotvec(np.array([0, 0, sensor2_rot]), degrees=True)

sensor3 = magpy.Sensor()
sensor3.position = (xs3, ys3, 0)
sensor3.orientation = R.from_rotvec(np.array([0, 0, sensor3_rot]), degrees=True)

if not load_from_csv:
    # not updated anymore since we will only use with loaded data
    vec_OP_base = np.array([distance_OP,0,0])
    vec_PM_base = np.array([xm,ym,offset_z])
    vec_PM_base2 = np.array([xm2,ym2,offset_z])

    cube = magpy.magnet.Cuboid(polarization=(Br, 0, 0),
                               dimension=(magnet_size_x, magnet_size_y, magnet_size_z))
    cube2 = magpy.magnet.Cuboid(polarization=(Br, 0, 0),
                               dimension=(magnet_size_x, magnet_size_y, magnet_size_z))

    angles_deg = np.linspace(start_angle, stop_angle, num_angle_steps)  # angle of second fingerpart

    ori = R.from_rotvec(np.array([(0, 0, t) for t in angles_deg]), degrees=True)
    ori_magnet = R.from_rotvec(np.array([(0, 0, t + base_rot_mag) for t in angles_deg]), degrees=True)
    ori_magnet2 = R.from_rotvec(np.array([(0, 0, t + base_rot_mag2) for t in angles_deg]), degrees=True)

    ori_half = R.from_rotvec(np.array([(0, 0, t / 2) for t in angles_deg]), degrees=True)
    pos_P = ori_half.apply(vec_OP_base)
    vec_PM = ori.apply(vec_PM_base)
    vec_PM2 = ori.apply(vec_PM_base2)

    vec_OM = pos_P + vec_PM
    vec_OM2 = pos_P + vec_PM2
    # Set path at initialization

    cube.orientation = ori_magnet
    cube.position = vec_OM

    cube2.orientation = ori_magnet2
    cube2.position = vec_OM2

    if second_magnet:
        collection = magpy.Collection(cube, cube2)
        B = magpy.getB(collection,sensor)*1e6 # convert to uT
        B2 = magpy.getB(collection,sensor2)*1e6
        B3 = magpy.getB(collection,sensor3)*1e6
    else:
        B = magpy.getB(cube,sensor)*1e6
        B2 = magpy.getB(cube,sensor2)* 1e6
        B3 = magpy.getB(cube,sensor3)* 1e6


else:
    X, angles_deg = load_data(CSV_LOAD_PATH)
    if double_outliers:
        X_extra = X[[0, -1], :]
        angles_extra = angles_deg[[0, -1]]
        X= np.vstack([X, X_extra])
        angles_deg = np.concatenate([angles_deg, angles_extra])

    B = X[:,0:3]
    B2 = X[:,3:6]
    if triple_sensor:
        B3 = X[:,6:9]


# overlap with training points and save training data to csv
headers = ['angle_joint', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2']
if triple_sensor:
    headers = ['angle_joint', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3']
sensor_center = (sensor.position + sensor2.position) / 2

B_data = []
for i in range(B.shape[0]):
    B_i = B[i]
    B2_i = B2[i]
    if triple_sensor: B3_i = B3[i]
    angle_joint = angles_deg[i]
    training_positions = get_n_points_in_shell(num_positions,min_training_radius,max_training_radius)

    if keepout_on_finger:
        # 1. Check if points are inside the bounds on ALL axes
        inside_mask = (training_positions[:, 0] > x_min-size_training_mag/2) & (training_positions[:, 0] < x_max+size_training_mag/2) & \
                      (training_positions[:, 1] > y_min-size_training_mag/2) & (training_positions[:, 1] < y_max+size_training_mag/2) & \
                      (training_positions[:, 2] > z_min-size_training_mag/2) & (training_positions[:, 2] < z_max+size_training_mag/2)
        training_positions = training_positions[~inside_mask]
        print(f"Points remaining: {len(training_positions)}")

    training_positions = training_positions + sensor_center
    cube3 = magpy.magnet.Cuboid(polarization=(Br, 0, 0),
                              dimension=(size_training_mag, size_training_mag, size_training_mag))
    cube3.position = training_positions
    cube3.orientation = R.random(num_positions)
    B_add = magpy.getB(cube3,sensor)*1e6
    B2_add = magpy.getB(cube3,sensor2)*1e6
    if triple_sensor: B3_add = magpy.getB(cube3,sensor3)*1e6

    B_new = B_i + B_add
    B2_new = B2_i + B2_add
    if triple_sensor: B3_new = B3_i + B3_add

    joint_angle_vec = angle_joint * np.ones((num_positions,1))
    if not triple_sensor:
        B_fused = np.hstack((joint_angle_vec, B_new, B2_new))
        B_fused_base = np.hstack((angle_joint,B_i, B2_i))
    else:
        B_fused = np.hstack((joint_angle_vec, B_new, B2_new, B3_new))
        B_fused_base = np.hstack((angle_joint, B_i, B2_i, B3_i))

    B_data.append([B_fused_base]*reps_of_baseline)
    B_data.append(B_fused)

B_data = np.vstack((B_data))

if add_noise:
    # Generate a multiplier with mean=1.0 and std_dev=0.05
    noise_multiplier = np.random.normal(loc=1.0, scale=noise_factor, size=B_data.shape)

    # Multiply the base data by the noise
    B_data = B_data * noise_multiplier



# 3. Write to file
# newline='' is important in Python 3 to prevent blank lines between rows
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row first
    writer.writerow(headers)

    # Write the data rows
    writer.writerows(B_data)

print(f"Successfully wrote data to {filename}")



