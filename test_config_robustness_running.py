import random
import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# Mock import for your simulation function
from triple_sensor_compare import simulate_config


USE_POLY = True
USE_ANGLE_ONLY = False
degree_poly = 3
def apply_block_shift(base_config, dx_m, dy_m, drot_deg):
    """
    Applies a rigid body transformation (block shift) to the magnet configuration.
    """
    xm, ym, rot, xm2, ym2, rot2 = base_config

    # Midpoint for rotation pivot
    cx = (xm + xm2) / 2.0
    cy = (ym + ym2) / 2.0

    drot_rad = math.radians(drot_deg)
    cos_val = math.cos(drot_rad)
    sin_val = math.sin(drot_rad)

    # Rotate around midpoint, then translate
    pxm = cx + (xm - cx) * cos_val - (ym - cy) * sin_val + dx_m
    pym = cy + (xm - cx) * sin_val + (ym - cy) * cos_val + dy_m
    prot = rot + drot_deg

    pxm2 = cx + (xm2 - cx) * cos_val - (ym2 - cy) * sin_val + dx_m
    pym2 = cy + (xm2 - cx) * sin_val + (ym2 - cy) * cos_val + dy_m
    prot2 = rot2 + drot_deg

    return (pxm, pym, prot, pxm2, pym2, prot2)


def extract_features(angles, mags):
    features = []
    for i in range(angles.shape[1]):
        ang_rad = np.deg2rad(angles[:, i])
        features.append(np.sin(ang_rad))
        features.append(np.cos(ang_rad))

        # Normalize magnitudes to keep feature scales balanced
        if not USE_ANGLE_ONLY:
            mag_max = np.max(mags[:, i]) if np.max(mags[:, i]) > 0 else 1.0
            features.append(mags[:, i] / mag_max)

    return np.column_stack(features)

def train_model(base_angles, base_mags, joint_angles, degree=degree_poly):
    X_train = extract_features(base_angles, base_mags)
    y_train = joint_angles

    # 3. Create and Train the Polynomial Model
    # A degree of 2 or 3 is usually optimal. Too high will cause extreme overfitting.
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    return model


def predict_joint_angles_poly(model,shifted_angles, shifted_mags):
    """
    Predicts the joint angle using a multivariable polynomial regression model
    trained on the continuous base calibration data.
    """
    X_test = extract_features(shifted_angles, shifted_mags)
    predicted_angles = model.predict(X_test)

    return predicted_angles

def predict_joint_angles(base_angles, base_mags, shifted_angles, shifted_mags, joint_angles):
    """
    Simulates the MCU logic: Predicts the joint angle by finding the closest
    match in the calibration data for each reading, with errors weighted
    by the magnetic field strength (magnitude) of the base sensors.
    """
    predicted = np.zeros(len(shifted_angles))

    # Normalize magnitudes so they don't overpower the angle differences
    mag_max = np.max(base_mags) if np.max(base_mags) > 0 else 1.0

    for i in range(len(shifted_angles)):
        # 1. Angular Difference (Wrap to [-180, 180] to handle phase jumps)
        ang_diff = (base_angles - shifted_angles[i] + 180) % 360 - 180

        # 2. Magnitude Difference (Scaled to degree-equivalents for fair weighting)
        mag_diff = ((base_mags - shifted_mags[i]) / mag_max) * 100

        # 3. Calculate raw squared errors per sensor
        if USE_ANGLE_ONLY:
            raw_error = ang_diff ** 2
        else:
            raw_error = ang_diff ** 2 + mag_diff ** 2

        # 4. Apply the magnitude weight
        # base_mags acts as a confidence multiplier for each sensor at that specific angle
        weighted_error = raw_error #* base_mags

        # 5. Total Cost (Sum of weighted errors across all sensors)
        cost = np.sum(weighted_error, axis=1)

        # 6. Find the calibration index with the lowest cost
        best_idx = np.argmin(cost)
        predicted[i] = joint_angles[best_idx]

    return predicted


def evaluate_shift_error(base_config, shifted_config, base_angles, base_mags, joint_angles, model=None):
    """
    Helper function: Simulates the shifted config, runs prediction, and returns errors.
    """
    # Simulate the slightly moved magnets
    shifted_angles, shifted_mags = simulate_config(*shifted_config)

    # Predict what the joint angle *appears* to be based on the shifted signals

    if USE_POLY:
        predicted_joints = predict_joint_angles_poly(model, shifted_angles,shifted_mags)
    else:
        predicted_joints = predict_joint_angles(base_angles, base_mags, shifted_angles, shifted_mags, joint_angles)

    # Calculate absolute errors
    errors = np.abs(predicted_joints - joint_angles)

    return np.max(errors), np.mean(errors)


def test_shift_tolerance(base_config, base_angles, base_mags, joint_angles,
                         max_dx_mm, max_dy_mm, max_drot_deg, num_iterations=100, avg_also=False):
    """
    Multi-iteration function: Fires random shifts within the limits and tracks the worst errors.
    """
    max_err_seen = 0.0
    avg_err_seen = 0.0
    avg_max_err = 0.0
    avg_avg_err = 0.0

    dx_m_limit = max_dx_mm * 1e-3
    dy_m_limit = max_dy_mm * 1e-3
    if USE_POLY:
        model = train_model(base_angles,base_mags,joint_angles)
    else:
        model = None

    for _ in range(num_iterations):
        # Generate random rigid body shift
        dx = random.uniform(-dx_m_limit, dx_m_limit)
        dy = random.uniform(-dy_m_limit, dy_m_limit)
        drot = random.uniform(-max_drot_deg, max_drot_deg)

        shifted_config = apply_block_shift(base_config, dx, dy, drot)

        cur_max_err, cur_avg_err = evaluate_shift_error(
            base_config, shifted_config, base_angles, base_mags, joint_angles, model=model,
        )

        # Track the absolute worst cases (they might come from different random iterations)
        if cur_max_err > max_err_seen:
            max_err_seen = cur_max_err
        if cur_avg_err > avg_err_seen:
            avg_err_seen = cur_avg_err
        avg_max_err += cur_max_err
        avg_avg_err += cur_avg_err

    avg_max_err /= num_iterations
    avg_avg_err /= num_iterations
    if avg_also:
        return max_err_seen, avg_err_seen, avg_max_err, avg_avg_err
    else:
        return max_err_seen, avg_err_seen


def run_robustness_sweep(base_config_mm_deg, joint_angles, num_iterations=200):
    """
    Main function: Loops over predefined tolerance steps and outputs a copy-pasteable block.
    """
    # 1. Unpack and convert base config to meters for the simulation
    xm_mm, ym_mm, rot_deg, xm2_mm, ym2_mm, rot2_deg = base_config_mm_deg
    base_config = (xm_mm * 1e-3, ym_mm * 1e-3, rot_deg, xm2_mm * 1e-3, ym2_mm * 1e-3, rot2_deg)

    # 2. Generate the Ground Truth (Calibration Data)
    print("Generating base calibration data...")
    base_angles, base_mags = simulate_config(*base_config)

    results = []
    print(f"Using Polynomial Regression: {USE_POLY}")
    print(f"Running sweep ({num_iterations} iterations per step)...\n")


    # 3. Loop over the tolerances (0.0 to 0.5 mm in 0.1 mm steps, 0 to 5° in 1° steps)
    for i in range(11):
        print(f"Run: {i}")
        shift_mm = i * 0.02
        shift_deg = i * 0.2

        if i == 0:
            name = "0.0 mm / 0.0° (Baseline)"
            # Baseline error should be 0, but good for a sanity check
            if USE_POLY:
                model = train_model(base_angles, base_mags, joint_angles)
            else:
                model = None
            max_err, avg_err = evaluate_shift_error(base_config, base_config, base_angles, base_mags, joint_angles, model=model)
            avg_max_err, avg_avg_err = max_err, avg_err
        else:
            name = f"{shift_mm:.1f} mm / {shift_deg:.1f}°"
            max_err, avg_err, avg_max_err, avg_avg_err = test_shift_tolerance(
                base_config, base_angles, base_mags, joint_angles,
                max_dx_mm=shift_mm, max_dy_mm=shift_mm, max_drot_deg=shift_deg,
                num_iterations=num_iterations, avg_also=True
            )

        # results.append((name, max_err, avg_err))
        results.append((name, max_err, avg_err, avg_max_err, avg_avg_err))

    # 4. Output Results
    print("--- COPY-PASTE BLOCK ---")
    print("Tolerance Limit, Worst Max Error [°], Worst Avg Error [°]")
    for name, max_err, avg_err, avg_max_err, avg_avg_err in results:
        print(f"{name}, {max_err:.4f}, {avg_err:.4f}, {avg_max_err:.4f}, {avg_avg_err:.4f}")
    print("------------------------\n")
    for name, max_err, avg_err, avg_max_err, avg_avg_err in results:
        print(f"{max_err:.4f}, {avg_err:.4f}, {avg_max_err:.4f}, {avg_avg_err:.4f}")
    print("------------------------\n")


if __name__ == "__main__":
    # Define the true joint angles your simulation sweeps over (e.g., 0 to 90 degrees in 91 steps)
    true_joint_angles = np.linspace(0, 90, 181)

    # Your best configuration in mm and degrees
    # my_best_config = (-7.299, -0.265, 348.933, -4.660, 5.713, 329.195)  # al
    my_best_config = (-4.217, 6.142, 109.416, -7.353, 0.504, 264.211)  # b2b

    run_robustness_sweep(
        base_config_mm_deg=my_best_config,
        joint_angles=true_joint_angles,
        num_iterations=1000
    )