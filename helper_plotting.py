import matplotlib.pyplot as plt
import numpy as np
import os


# with constraints
def plot_convergence(res):
    n_evals = []  # Number of evaluations
    hist_F = []  # Objective Value (Score)
    hist_CV = []  # Constraint Violation

    # Iterate through the saved history
    for algorithm in res.history:
        n_evals.append(algorithm.n_gen)

        # Get the best individual of this generation
        opt = algorithm.opt[0]

        # Store data
        # Note: We store -opt.F[0] because we want the Positive Score
        hist_F.append(-opt.F[0])
        hist_CV.append(opt.CV[0])

    # --- PLOT SETUP ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot 1: The Score (Blue Line)
    color = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Objective Score (Higher is Better)', color=color)
    ax1.plot(n_evals, hist_F, color=color, linewidth=2, label="Best Score")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Constraint Violation (Red Dashed Line)
    # This helps you see WHEN the solution became valid.
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Constraint Violation (0 = Valid)', color=color)
    ax2.plot(n_evals, hist_CV, color=color, linestyle='--', alpha=0.5, label="Invalidity")
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight the "Feasible" zone
    # If CV drops to 0, that's when real optimization started
    valid_indices = [i for i, cv in enumerate(hist_CV) if cv <= 1e-9]
    if valid_indices:
        first_valid_gen = n_evals[valid_indices[0]]
        plt.axvline(x=first_valid_gen, color='green', linestyle=':', label='Feasible Region Found')
        plt.text(first_valid_gen, plt.ylim()[1] * 0.95, ' Valid', color='green', fontweight='bold')

    plt.title("Convergence History: Score vs. Constraints")
    plt.tight_layout()
    plt.show()


# without constraints
def plot_score_evolution(res):
    """
    Plots only the evolution of the Best Score over generations.
    """
    if not res.history:
        print("Error: No history found. Make sure you set 'save_history=True' in minimize().")
        return

    n_gens = []
    best_scores = []

    # Extract data from Pymoo history
    for algorithm in res.history:
        n_gens.append(algorithm.n_gen)

        # Get the best individual of this generation
        opt = algorithm.opt[0]

        # Invert the score back to positive (since Pymoo minimized the negative)
        # If your problem was naturally minimizing, remove the negative sign.
        score = -opt.F[0]
        best_scores.append(score)

    # --- PLOT ---
    plt.figure(figsize=(10, 6))

    plt.plot(n_gens, best_scores,
             color='#0072B2',  # Professional Blue
             linewidth=2.5,
             linestyle='-',
             marker='o',
             markersize=4,
             alpha=0.8,
             label="Best Score")

    # Add a "Peak" marker at the very end
    final_score = best_scores[-1]
    plt.scatter(n_gens[-1], final_score, color='red', s=100, zorder=5, label=f"Final: {final_score:.4f}")

    plt.title(f'Optimization for magnet placement', fontsize=14, fontweight='bold')
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.show()




def plot_magnetic_field(angles, magnitudes, x_axis=None, x_range=(0, 90),
                        num_sensors=None, as_polar=False, show_range=False,
                        unwrap_phase=True):
    """
    Plots magnetic field magnitudes and angles for multiple sensors.

    Parameters:
    - angles (np.ndarray): 2D array of angles in degrees, shape (steps, sensors).
    - magnitudes (np.ndarray): 2D array of field magnitudes, shape (steps, sensors).
    - x_axis (np.ndarray, optional): Custom 1D array for the X-axis.
    - x_range (tuple, optional): (min, max) degrees for the X-axis sweep. Defaults to (0, 90).
    - num_sensors (int, optional): Number of sensors to plot. Defaults to all available.
    - as_polar (bool): If True, generates a polar plot.
    - show_range (bool): If True, calculates max-min angle and displays it in the legend.
    - unwrap_phase (bool): If True, corrects for -180/180 phase jumps before calculating/plotting.
    """
    # Infer dimensions
    total_steps, total_sensors = angles.shape

    # Handle optional inputs
    if num_sensors is None:
        num_sensors = total_sensors
    else:
        num_sensors = min(num_sensors, total_sensors)

    # Map steps to the specified degree range if a custom x_axis isn't provided
    if x_axis is None:
        x_axis = np.linspace(x_range[0], x_range[1], total_steps)

    colors = plt.cm.tab10.colors

    if as_polar:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        for i in range(num_sensors):
            color = colors[i % len(colors)]

            # Phase unwrapping (requires converting to radians and back)
            if unwrap_phase:
                rad_angles = np.unwrap(np.deg2rad(angles[:, i]))
            else:
                rad_angles = np.deg2rad(angles[:, i])

            mags = magnitudes[:, i]

            # Create legend label
            label = f'Sensor {i + 1}'
            if show_range:
                angle_diff = np.rad2deg(np.max(rad_angles) - np.min(rad_angles))
                label += f' (Δ: {angle_diff:.1f}°)'

            # Plot the main trace
            line, = ax.plot(rad_angles, mags, color=color, linewidth=2, label=label)

            # Mark the Start Point
            ax.plot(rad_angles[0], mags[0], marker='o', color=color, markersize=8,
                    markeredgecolor='black')

            # Add a Direction Arrow at the very end of the trace
            if total_steps > 1:
                ax.annotate('', xy=(rad_angles[-1], mags[-1]),
                            xytext=(rad_angles[-2], mags[-2]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2))

        ax.set_title("Magnetic Field Vector Trace\n(Circles mark the start point)", va='bottom', fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for i in range(num_sensors):
            color = colors[i % len(colors)]

            # Phase unwrapping for linear plot
            if unwrap_phase:
                plot_angles = np.rad2deg(np.unwrap(np.deg2rad(angles[:, i])))
            else:
                plot_angles = angles[:, i]

            # Create legend label
            label = f'Sensor {i + 1}'
            if show_range:
                angle_diff = np.max(plot_angles) - np.min(plot_angles)
                label += f' (Δ: {angle_diff:.1f}°)'

            # Magnitude Plot
            ax1.plot(x_axis, magnitudes[:, i], label=label, color=color, linewidth=2)


            # Angle Plot
            ax2.plot(x_axis, plot_angles, label=label, color=color, linewidth=2)

        # Formatting Magnitudes
        ax1.set_ylim(bottom=0)
        ax1.set_ylabel('Magnitude [mT]', fontsize=12)
        ax1.set_title('Magnetic Field Magnitude and Angle', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # Clamp X-Axis
        ax1.set_xlim(x_range[0], x_range[1])

        # Formatting Angles
        ax2.set_ylabel('Angle [°]', fontsize=12)
        ax2.set_xlabel(f'Joint Angle [°]', fontsize=12)

        # Removed the manual y-limits/yticks so matplotlib dynamically scales
        # to the exact band of your unwrapped data.
        ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_sensor_information(angles, magnitudes, x_range=(0, 90), num_sensors=None):
    """
    Plots the information content (weighted angular difference) of each sensor
    over the specified joint angle range, alongside the total sum.

    Parameters:
    - angles (np.ndarray): 2D array of magnetic angles in degrees, shape (steps, sensors).
    - magnitudes (np.ndarray): 2D array of field magnitudes, shape (steps, sensors).
    - x_range (tuple, optional): (min, max) joint angles for the sweep. Defaults to (0, 90).
    - num_sensors (int, optional): Number of sensors to plot. Defaults to all available.
    """
    total_steps, total_sensors = angles.shape

    if num_sensors is None:
        num_sensors = total_sensors
    else:
        num_sensors = min(num_sensors, total_sensors)

    magnitudes = magnitudes * 1000  # convert to micro T from mT
    num_angle_steps = len(angles[:,0])
    angle_scale = num_angle_steps / (x_range[1]-x_range[0])
    # 1. Calculate the Information Metric
    diffs = np.diff(angles, axis=0)
    diffs = (diffs + 180) % 360 - 180  # Wrap to [-180, 180]

    mag_avgs = (magnitudes[:-1, :] + magnitudes[1:, :]) / 2
    weighted_angle_diffs = np.abs(np.deg2rad(diffs) * mag_avgs) * angle_scale

    # Calculate the total information (sum across the selected sensors)
    total_info = np.sum(weighted_angle_diffs[:, :num_sensors], axis=1)

    # 2. Calculate X-axis midpoints
    # Since np.diff reduces the length by 1, we map the results to the midpoints
    # of the joint angle steps to keep the X-axis perfectly aligned physically.
    joint_angles = np.linspace(x_range[0], x_range[1], total_steps)
    plot_x = (joint_angles[:-1] + joint_angles[1:]) / 2

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    # colors = ["cyan","magenta","orange"]

    # Plot individual sensors
    for i in range(num_sensors):
        color = colors[i % len(colors)]
        ax.plot(plot_x, weighted_angle_diffs[:, i], label=f'Sensor {i + 1}',
                color=color, linewidth=2, alpha=0.8)

    # Plot the total sum as a prominent dashed line
    ax.plot(plot_x, total_info, label='Total Information (Sum)',
            color='black', linewidth=3, linestyle='--')

    # Formatting
    ax.set_title('Sensor Information Map over Joint Angle', fontsize=14)
    ax.set_xlabel(f'Joint Angle [°]', fontsize=12)
    ax.set_ylabel('Signal Information\n(|Δθ| × Avg Magnitude) [µT]', fontsize=12)
    ax.set_xlim(x_range[0], x_range[1])

    # Start y-axis at 0 to accurately reflect lack of information
    ax.set_ylim(bottom=0)

    ax.grid(True, linestyle='--', alpha=0.7)
    #ax.legend(loc='best')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_combined_pareto(res_standard, res_forced_b2b):
    plt.figure(figsize=(10, 6))

    # --- 1. Plot Standard Run (Aligned) ---
    if res_standard.F is not None:
        # Force 2D array to safely slice columns
        F1 = np.atleast_2d(res_standard.F)

        # Invert scores back to positive: -F[:, 0] and -F[:, 1]
        plt.scatter(-F1[:, 0], -F1[:, 1],
                    color='tab:blue',
                    s=50,
                    alpha=0.7,
                    label='Run 1: Standard Search')

    # --- 2. Plot Forced Back-to-Back Run ---
    if res_forced_b2b.F is not None:
        F2 = np.atleast_2d(res_forced_b2b.F)

        plt.scatter(-F2[:, 0], -F2[:, 1],
                    color='tab:red',
                    marker='X',  # Different shape to make it distinct
                    s=60,
                    alpha=0.8,
                    label='Run 2: Forced Back-to-Back')

    # --- 3. Formatting ---
    plt.title("Pareto Front: Standard vs. Back-to-Back", fontsize=14, fontweight='bold')
    plt.xlabel("Score min", fontsize=12)
    plt.ylabel("Score average", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.tight_layout()

    plt.show()


def plot_combined_pareto_with_tradeoff(file_aligned, file_b2b, weight_obj1=1.0, weight_obj2=0.3, show_opt=True):
    plt.figure(figsize=(10, 6))

    best_overall_score = -np.inf
    best_point = None

    # --- 1. Load and Plot Run 1 (Aligned) ---
    if os.path.exists(file_aligned):
        data1 = np.loadtxt(file_aligned, delimiter=',', skiprows=1)
        data1 = np.atleast_2d(data1)

        x1 = data1[:, 0]
        y1 = data1[:, 1]

        plt.scatter(x1, y1, color='tab:blue', s=50, alpha=0.7, label='Run 1: Standard Search')

        # Find best point in Run 1
        scores1 = (weight_obj1 * x1) + (weight_obj2 * y1)
        idx1 = np.argmax(scores1)
        if scores1[idx1] > best_overall_score:
            best_overall_score = scores1[idx1]
            best_point = (x1[idx1], y1[idx1])

    else:
        print(f"File not found: {file_aligned}")

    # --- 2. Load and Plot Run 2 (Back-to-Back) ---
    if file_b2b is None:
        best_config = data1[idx1, 2:]
        print("Best config:")
        print(", ".join(f"{val * 1000 if (i % 3) != 2 else val:.3f}" for i, val in enumerate(best_config)))
    elif os.path.exists(file_b2b):
        data2 = np.loadtxt(file_b2b, delimiter=',', skiprows=1)
        data2 = np.atleast_2d(data2)

        x2 = data2[:, 0]
        y2 = data2[:, 1]

        plt.scatter(x2, y2, color='tab:red', marker='X', s=60, alpha=0.8, label='Run 2: Forced Back-to-Back')

        # Find best point in Run 2
        scores2 = (weight_obj1 * x2) + (weight_obj2 * y2)
        idx2 = np.argmax(scores2)
        if scores2[idx2] > best_overall_score:
            best_overall_score = scores2[idx2]
            best_point = (x2[idx2], y2[idx2])
            best_config = data2[idx2,2:]
        else:
            best_config = data1[idx1,2:]
        print("Best config:")
        print(", ".join(f"{val * 1000 if (i % 3) != 2 else val:.3f}" for i, val in enumerate(best_config)))

    else:
        print(f"File not found: {file_b2b}")

    # --- 3. Plot the Trade-off Isoline ---
    if best_point is not None and show_opt:
        best_x, best_y = best_point
        print(f"Best scores (min,avg, {weight_obj1}*min+{weight_obj2}*avg): {best_x:.2f}, {best_y:.2f}, "
              f"{(best_x*weight_obj1+best_y*weight_obj2):.2f}")

        # Highlight the absolute best point
        plt.scatter(best_x, best_y, color='gold', edgecolors='black', marker='*', s=300, zorder=5,
                    label=f'Best Trade-off (Score: {best_overall_score:.2f})')

        # Draw the tangent line
        # We grab the current auto-scaled limits so the line fits perfectly
        ax = plt.gca()
        x_vals = np.array(ax.get_xlim())
        y_limits = ax.get_ylim()  # Save original Y limits

        # Calculate Y values for the line: Y = (Best_Score - W1*X) / W2
        y_vals = (best_overall_score - (weight_obj1 * x_vals)) / weight_obj2

        plt.plot(x_vals, y_vals, color='gold', linestyle='--', linewidth=2, zorder=4,
                 label=f'Trade-off Slope ({weight_obj1}x + {weight_obj2}y)')

        # Re-apply the original Y limits so the line doesn't distort the graph vertically
        ax.set_ylim(y_limits)

    # --- 4. Formatting ---
    if file_b2b is None:
        plt.title("Pareto Front", fontsize=14, fontweight='bold')
    else:
        plt.title("Pareto Front: Standard vs. Back-to-Back", fontsize=14, fontweight='bold')
    plt.xlabel("Score min", fontsize=12)
    plt.ylabel("Score average", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.show()


def compare_sensor_information(angles1, magnitudes1, angles2, magnitudes2,
                               label1="Config 1", label2="Config 2",
                               x_range=(0, 90), mode='all'):
    """
    Compares the total information content of two different sensor configurations.

    Parameters:
    - angles1, magnitudes1: Arrays for the first configuration.
    - angles2, magnitudes2: Arrays for the second configuration.
    - label1, label2: String labels for the legend.
    - x_range: (min, max) joint angles for the sweep. Defaults to (0, 90).
    - mode: 'both' (plots sums), 'diff' (plots difference), or 'all' (plots sums and difference).
    """

    # Helper function to calculate the total sum to avoid repeating code
    def _calc_total_info(angles, magnitudes):
        num_angle_steps = len(angles[:, 0])
        angle_scale = num_angle_steps / (x_range[1] - x_range[0])

        mags_uT = magnitudes * 1000  # convert to micro T from mT

        diffs = np.diff(angles, axis=0)
        diffs = (diffs + 180) % 360 - 180  # Wrap to [-180, 180]

        mag_avgs = (mags_uT[:-1, :] + mags_uT[1:, :]) / 2
        weighted_angle_diffs = np.abs(np.deg2rad(diffs) * mag_avgs) * angle_scale

        return np.sum(weighted_angle_diffs, axis=1)

    # 1. Calculate Information for both sets
    info1 = _calc_total_info(angles1, magnitudes1)
    info2 = _calc_total_info(angles2, magnitudes2)

    # Ensure arrays match in length (in case of different step resolutions)
    min_len = min(len(info1), len(info2))
    info1 = info1[:min_len]
    info2 = info2[:min_len]
    info_diff = info1 - info2

    # 2. Calculate X-axis midpoints
    total_steps = len(info1) + 1
    joint_angles = np.linspace(x_range[0], x_range[1], total_steps)
    plot_x = (joint_angles[:-1] + joint_angles[1:]) / 2

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    if mode in ['both', 'all']:
        ax.plot(plot_x, info1, label=f'Total Info: {label1}', color='tab:blue', linewidth=2.5)
        ax.plot(plot_x, info2, label=f'Total Info: {label2}', color='tab:orange', linewidth=2.5)

    if mode in ['diff', 'all']:
        if mode == 'all':
            # Create a secondary y-axis for the difference line
            ax_diff = ax.twinx()
            ax_diff.plot(plot_x, info_diff, label=f'Difference ({label1} - {label2})',
                         color='tab:red', linewidth=2, linestyle=':')
            ax_diff.set_ylabel('Difference [µT]', fontsize=12, color='tab:red')
            ax_diff.tick_params(axis='y', labelcolor='tab:red')
        else:
            # If ONLY plotting difference, use the main axis
            ax.plot(plot_x, info_diff, label=f'Difference ({label1} - {label2})',
                    color='tab:red', linewidth=2.5)

    # Formatting
    ax.set_title('Sensor Information Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Joint Angle [°]', fontsize=12)
    ax.set_xlim(x_range[0], x_range[1])
    ax.grid(True, linestyle='--', alpha=0.7)

    if mode != 'diff':
        ax.set_ylabel('Total Signal Information [µT]', fontsize=12)
        ax.set_ylim(bottom=0)

    # Handle legends based on whether we used the twin axis
    if mode == 'all':
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_diff.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
    else:
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()