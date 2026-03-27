import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
from matplotlib.patches import FancyArrow
import time

# --- 1. CONFIGURATION ---
# Magnet Dimensions (m)
cube_size = 3
cube_dim = (cube_size, cube_size, cube_size)

# Polarization (mT)
Br = 1345
mag_pol = (Br, 0, 0)

# Fixed Original Sensor Position (m -> mm)
# DO NOT CHANGE: Keeping this anchors the plot's (0,0) so image alignment won't break.
xsens = 0.00701
ysens = 0.00188
sensor_pos = (xsens * 1e3, ysens * 1e3, 0)

# NEW SENSORS
x_s1, y_s1 = 0.006925, -0.000589
x_s2, y_s2 = 0.005681, 0.003913
x_s3, y_s3 = 0.0025, 0.000268

# Calculate their relative positions on the plot (in mm) relative to the plot's (0,0)
p_s1 = (x_s1 * 1e3 - sensor_pos[0], y_s1 * 1e3 - sensor_pos[1], 0)
p_s2 = (x_s2 * 1e3 - sensor_pos[0], y_s2 * 1e3 - sensor_pos[1], 0)
p_s3 = (x_s3 * 1e3 - sensor_pos[0], y_s3 * 1e3 - sensor_pos[1], 0)

# Magnet Positions (mm)
#xm, ym, base_rot_mag, xm2, ym2, base_rot_mag2 = -7.299, -0.265, 348.933, -4.660, 5.713, 329.195
xm, ym, base_rot_mag, xm2, ym2, base_rot_mag2 = -4.217, 6.142, 109.416, -7.353, 0.504, 264.211



if 0:
    base_rot_mag-=180
    base_rot_mag2-=180

single_magnet_only = False
if single_magnet_only:
    xm2, ym2, base_rot_mag2 = 1e4, 1e4, 150.

# Joint setup (mm)
distance_OP = 20
offset_z = 0

# Animation settings
deg_step = 1
fancy_colors = False
background_images = True
PATH_STATIC = "images/stator2.png"
PATH_ROTOR = "images/rotor_clean.png"

# Grid for field visualization
grid_basic = False
if grid_basic:
    grid_range = 17
    grid_min = -grid_range
    grid_miny = -grid_range
    grid_max = grid_range
    grid_maxy = grid_range
else:
    grid_min = -6
    grid_miny = -6
    grid_max = 12
    grid_maxy = 12

grid_step = 1.1
xs = np.arange(grid_min, grid_max + 1, grid_step)
ys = np.arange(grid_miny, grid_maxy + 1, grid_step)
X, Y = np.meshgrid(xs, ys)
grid_coords = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))

# --- 2. SETUP OBJECTS ---
mag1 = magpy.magnet.Cuboid(polarization=mag_pol, dimension=cube_dim, position=(xm, ym, 0))
mag2 = magpy.magnet.Cuboid(polarization=mag_pol, dimension=cube_dim, position=(xm2, ym2, 0))

coll = magpy.Collection(mag1, mag2)

# --- 3. VISUALIZATION SETUP ---
fig, ax = plt.subplots(figsize=(9, 9))
plt.subplots_adjust(bottom=0.2)

if background_images:
    try:
        # 1. STATIONARY IMAGE
        img_raw = plt.imread(PATH_STATIC)
        img_rotated = np.rot90(img_raw, k=1)
        desired_center_x = -17.4
        desired_center_y = 1.8
        target_height = 13.5
        r_h, r_w = img_rotated.shape[:2]
        aspect_ratio = r_h / r_w
        target_width = target_height / aspect_ratio
        half_w = target_width / 2.0
        half_h = target_height / 2.0
        new_extent = [desired_center_x - half_w, desired_center_x + half_w,
                      desired_center_y - half_h, desired_center_y + half_h]
        ax.imshow(img_rotated, extent=new_extent, zorder=0, alpha=0.3)

        # 2. MOVING IMAGE
        img_mov_raw = plt.imread(PATH_ROTOR)
        img_mov_rotated = np.rot90(img_mov_raw, k=1)
        target_w_rotor = 13.7
        r_h, r_w = img_mov_rotated.shape[:2]
        aspect_ratio_mov = r_h / r_w
        target_h_rotor = target_w_rotor * aspect_ratio_mov
        rotor_extent = [-target_w_rotor / 2, target_w_rotor / 2,
                        -target_h_rotor / 2, target_h_rotor / 2]
        im_rotor = ax.imshow(img_mov_rotated, extent=rotor_extent, zorder=2, alpha=0.4)

    except Exception as e:
        print(f"Image Error: {e}")
        im_rotor = None

cmap = plt.cm.viridis
norm = mcolors.LogNorm(vmin=10, vmax=500)

# Background Field Quiver
quiver = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y),
                   pivot='mid', scale=24, width=0.0045)

rect1 = Rectangle((0, 0), cube_dim[0], cube_dim[1], color='navy', label='Mag 1', alpha=0.6)
rect2 = Rectangle((0, 0), cube_dim[0], cube_dim[1], color='darkgreen', label='Mag 2', alpha=0.6)

rect1.set_bounds(-cube_dim[0] / 2, -cube_dim[1] / 2, cube_dim[0], cube_dim[1])
rect2.set_bounds(-cube_dim[0] / 2, -cube_dim[1] / 2, cube_dim[0], cube_dim[1])

ax.add_patch(rect1)
ax.add_patch(rect2)

arrow_len = cube_dim[0] * 0.8
arrow_start = -arrow_len / 2

arrow1 = FancyArrow(x=arrow_start, y=0, dx=arrow_len, dy=0,
                    width=cube_dim[1] * 0.08, length_includes_head=True,
                    head_width=cube_dim[1] * 0.26, color='navy', zorder=6, alpha=1)

arrow2 = FancyArrow(x=arrow_start, y=0, dx=arrow_len, dy=0,
                    width=cube_dim[1] * 0.08, length_includes_head=True,
                    head_width=cube_dim[1] * 0.26, color='darkgreen', zorder=6, alpha=1)

ax.add_patch(arrow1)
ax.add_patch(arrow2)

colors = plt.cm.tab10.colors

# Plot the Sensor Points (Blue, Orange, Green)
ax.plot(p_s1[0], p_s1[1], marker='o', markersize=6, color=colors[0], markeredgecolor='black', zorder=10)
ax.plot(p_s2[0], p_s2[1], marker='o', markersize=6, color=colors[1], markeredgecolor='black', zorder=10)
ax.plot(p_s3[0], p_s3[1], marker='o', markersize=6, color=colors[2], markeredgecolor='black', zorder=10)

ax.text(p_s1[0] + 0.5, p_s1[1] + 0.5, 'S1', color=colors[0], fontweight='bold', zorder=10)
ax.text(p_s2[0] + 0.5, p_s2[1] + 0.5, 'S2', color=colors[1], fontweight='bold', zorder=10)
ax.text(p_s3[0] + 0.5, p_s3[1] + 0.5, 'S3', color=colors[2], fontweight='bold', zorder=10)

# Setup independent Quiver for the Sensors (Arrows tracking field at exact sensor points)
sensor_xs = [p_s1[0], p_s2[0], p_s3[0]]
sensor_ys = [p_s1[1], p_s2[1], p_s3[1]]
sensor_quiver = ax.quiver(sensor_xs, sensor_ys, [1, 1, 1], [0, 0, 0],
                          pivot='mid', scale=15, width=0.007,
                          color=[colors[0], colors[1], colors[2]],
                          edgecolors='black', linewidths=0.5, zorder=12)

status_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round", facecolor='white', alpha=0.9), zorder=20)

ax.set_xlim(grid_min, grid_max)
ax.set_ylim(grid_miny, grid_maxy)
ax.set_aspect('equal')
ax.set_title("Magnetic Field Simulation")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.grid(True, linestyle='--', alpha=0.3)


# --- 4. ANIMATION UPDATE FUNCTION ---
def update(frame):
    angle = frame * deg_step
    angle_rad = np.deg2rad(angle)

    ori1 = R.from_rotvec(np.array([0, 0, base_rot_mag + angle]), degrees=True)
    ori2 = R.from_rotvec(np.array([0, 0, base_rot_mag2 + angle]), degrees=True)

    ori = R.from_rotvec(np.array([0, 0, angle]), degrees=True)
    ori_half = R.from_rotvec(np.array([0, 0, angle / 2]), degrees=True)

    vec_OP_base = np.array([distance_OP, 0, 0])
    vec_PM_base = np.array([xm, ym, offset_z])
    vec_PM_base2 = np.array([xm2, ym2, offset_z])

    pos_P = ori_half.apply(vec_OP_base)
    vec_PM = ori.apply(vec_PM_base)
    vec_PM2 = ori.apply(vec_PM_base2)

    vec_OM = pos_P + vec_PM
    vec_OM2 = pos_P + vec_PM2

    vec_OS = np.array(sensor_pos)
    vec_SM = vec_OM - vec_OS
    vec_SM2 = vec_OM2 - vec_OS

    mag1.position = vec_SM
    mag1.orientation = ori1
    mag2.position = vec_SM2
    mag2.orientation = ori2

    angle1_rad = np.deg2rad(base_rot_mag + angle)
    angle2_rad = np.deg2rad(base_rot_mag2 + angle)

    if background_images:
        if im_rotor is not None:
            pivot_offset_x = 3.73
            pivot_offset_y = -1.75

            P_plot_x = pos_P[0] - vec_OS[0]
            P_plot_y = pos_P[1] - vec_OS[1]

            t_img = (transforms.Affine2D()
                     .translate(-pivot_offset_x, -pivot_offset_y)
                     .rotate(angle_rad)
                     .translate(P_plot_x, P_plot_y)
                     + ax.transData)

            im_rotor.set_transform(t_img)

    t1 = (plt.matplotlib.transforms.Affine2D()
          .rotate(angle1_rad)
          .translate(vec_SM[0], vec_SM[1]) + ax.transData)

    t2 = (plt.matplotlib.transforms.Affine2D()
          .rotate(angle2_rad)
          .translate(vec_SM2[0], vec_SM2[1]) + ax.transData)

    rect1.set_transform(t1)
    rect2.set_transform(t2)
    arrow1.set_transform(t1)
    arrow2.set_transform(t2)

    # --- Update Background Field ---
    B_grid = coll.getB(grid_coords)
    Bx = B_grid[:, 0]
    By = B_grid[:, 1]

    B_mag_grid = np.linalg.norm(B_grid, axis=1)
    B_mag_safe = np.copy(B_mag_grid)
    B_mag_safe[B_mag_safe == 0] = 1

    quiver.set_UVC(Bx / B_mag_safe, By / B_mag_safe)
    colors = cmap(norm(B_mag_grid))

    if fancy_colors:
        quiver.set_facecolor(colors)
    else:
        colors_blanc = np.zeros_like(colors) + [0.5, 0.5, 0.5, 0.6]
        quiver.set_facecolor(colors_blanc)

    # --- Update Sensor Readings and Arrows ---
    B_s1 = coll.getB(p_s1)
    B_s2 = coll.getB(p_s2)
    B_s3 = coll.getB(p_s3)

    mag1_val = np.linalg.norm(B_s1)
    mag2_val = np.linalg.norm(B_s2)
    mag3_val = np.linalg.norm(B_s3)

    ang1_val = np.degrees(np.arctan2(B_s1[1], B_s1[0]))
    ang2_val = np.degrees(np.arctan2(B_s2[1], B_s2[0]))
    ang3_val = np.degrees(np.arctan2(B_s3[1], B_s3[0]))

    # Combine for vectorized quiver update
    B_sensors = np.array([B_s1, B_s2, B_s3])
    B_mag_sensors = np.array([mag1_val, mag2_val, mag3_val])
    B_mag_sensors[B_mag_sensors == 0] = 1  # Safe div

    sensor_quiver.set_UVC(B_sensors[:, 0] / B_mag_sensors, B_sensors[:, 1] / B_mag_sensors)

    status_text.set_text(
        f"Joint Angle: {angle:.0f}°\n"
        f"--------------------------\n"
        f"S1 (Cyan):  {mag1_val:5.1f} mT, {ang1_val:6.1f}°\n"
        f"S2 (Mag):   {mag2_val:5.1f} mT, {ang2_val:6.1f}°\n"
        f"S3 (Org):   {mag3_val:5.1f} mT, {ang3_val:6.1f}°"
    )

    return quiver, rect1, rect2, arrow1, arrow2, sensor_quiver, status_text


# --- 5. PAUSE BUTTON SETUP ---
anim_running = True


def toggle_pause(event):
    global anim_running
    if anim_running:
        anim.event_source.stop()
        button.label.set_text("Play")
    else:
        anim.event_source.start()
        button.label.set_text("Pause")
    anim_running = not anim_running


ax_button = plt.axes([0.45, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Pause')
button.on_clicked(toggle_pause)

# --- 6. RUN ---
anim = animation.FuncAnimation(fig, update, frames=int(91 / deg_step), interval=70, blit=False)
plt.show()