import time
import serial
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import torch
import joblib
import math
import csv
import keyboard

import matplotlib.pyplot as plt
from collections import deque
import time


from train_angle_model import MLPRegressor   # same class definition as used during training

"""
Takes Live Data from Teensy, Loads Trained NN model and previously recorded data
Then tries to predict what the joint angle is
Different prediction methods can be turned on through flags
Features for NN have to be set correctly, need to be the same as during training

If logging is enabled, allows to log to a csv file upon pressing L
"""

# -----------------------------
# Config
# -----------------------------
NAME = "sim_pcb2_exp3_2_1_1" # set NN name here, on all 3 sensors
NAME2 = "sim_pcb2_exp3_2_1_2"  # set NN2 name here, only on 2 sensors

#CLASSIC_DATA_PATH = "classic_data/pcb2_mcp_exp2_2deg.csv"
CLASSIC_DATA_PATH = "logs/PCB_v2/Experiment 3/mcp_2.csv"

# -- LOGGING
logging_enabled = True  # if true logs data on L
LOG_FILE = 'logs/PCB_v2/Experiment 3/predict_crosstalk_60.csv'
joint_angle = 60  # if logging at fixed angle, added

# set features same as during training
feature_xy = True
feature_z = True
feature_magnitude = True
feature_angle = True
feature_diff_xy = True
feature_diff_mag = True
feature_dot_cross = True

NUM_SENSORS = 3 # max available is 3 for MCP, 2 for PIP



second_net_double_sensor = True # only front 2 instead of 3 sensors used

# ---- SELECT PREDICTION METHODS ----
use_nn = True
use_nn2 = True
use_classic = False # directly predicts for each sensor separately only on angle
use_diff = False
use_poly = True # poly triple 2
use_poly2 = True # poly triple 3
use_poly3 = True # poly double 2
use_poly4= True # poly double 3
intervall = 550 # interval between two predictions in ms

plot_output = False

shift_angle_2 = False # %360+180 second angle because over jump



MODEL_PATH = "models/"+NAME+"/model.pth"
ARCHITECTURE_PATH = "models/"+NAME+"/model_config.pth"
SCALER_PATH = "models/"+NAME+"/scalar.joblib"

MODEL2_PATH = "models/"+NAME2+"/model.pth"
ARCHITECTURE2_PATH = "models/"+NAME2+"/model_config.pth"
SCALER2_PATH = "models/"+NAME2+"/scalar.joblib"


SERIAL_PORT = "COM7"         # adapt to your Teensy port
BAUDRATE = 115200
MC_DROPOUT_SAMPLES = 0       # number of forward passes for uncertainty, set to 0 for no dropout, no uncertainty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Expected input fields in order
FIELDS = ["x_1","y_1","z_1","x_2","y_2","z_2","x_3","y_3","z_3"]
number_of_sensors = NUM_SENSORS




# Definitions - NOT config
models = []
models2 = []
model_diff = None
poly_model = None
poly_model2 = None
poly_model3 = None
poly_model4 = None

poly = PolynomialFeatures(degree=3)
log_requested = False

# -----------------------------
# Utility: functions
# -----------------------------
class LivePlotter:
    def __init__(self, window=200):
        self.window = window
        self.data = {}
        self.fig, self.ax = plt.subplots()
        plt.ion()
        self.lines = {}

    def update(self, predictions):
        for d in predictions:
            key = d["type"]
            val = d["value"]

            if key not in self.data:
                self.data[key] = deque(maxlen=self.window)
                self.lines[key], = self.ax.plot([], [], label=key)
                self.ax.legend(loc="upper left")

            self.data[key].append(val)
            self.lines[key].set_data(range(len(self.data[key])), self.data[key])

        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)



def request_log_entry(event):
    """Callback: simply sets the flag when L is pressed."""
    global log_requested
    if logging_enabled:
        log_requested = True

def enable_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


# -----------------------------
# Load model + scaler
# -----------------------------
def load_model_and_scaler(net: int):
    if net == 1:
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded.")
        architecture = torch.load(ARCHITECTURE_PATH)
        print(architecture)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
    elif net == 2:
        scaler = joblib.load(SCALER2_PATH)
        print("Scaler loaded.")
        architecture = torch.load(ARCHITECTURE2_PATH)
        print(architecture)
        state = torch.load(MODEL2_PATH, map_location=DEVICE)
    else:
        print("Error net not known")

    model = MLPRegressor(input_dim=architecture['input_dim'], hidden_dims=architecture['hidden_dims'], use_batchnorm=architecture['use_batchnorm']) #adjust input dim here if necessary
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    return model, scaler


# -----------------------------
# Inference with MC-Dropout
# -----------------------------
def predict_with_uncertainty(model, x_scaled, mc_samples=30):
    """
    Perform MC-dropout by running forward pass multiple times.
    Returns (mean_prediction, uncertainty_std)
    """
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
    preds = []
    if mc_samples:
        enable_dropout(model)   # important: keeps dropout active

        with torch.no_grad():
            for _ in range(mc_samples):
                y = model(x_tensor).cpu().numpy().reshape(-1)
                preds.append(y)

    else:
        with torch.no_grad():
            y = model(x_tensor).cpu().numpy().reshape(-1)
            preds.append(y)

    preds = np.array(preds).reshape(-1) * 90 # normalize back to 0-90°
    mean = preds.mean()
    std = preds.std()  # uncertainty estimate

    return mean, std


# -----------------------------
# Parse Teensy input line
# -----------------------------
def parse_teensy_line(line: str):
    """
    Expects a comma-separated string (e.g., "x_1,y_1,z_1,x_2,y_2,z_2...")
    Dynamically scales for 1, 2, or 3 sensors.
    Returns numpy array of shape (1, num_features)
    """
    parts = [f.strip() for f in line.split(',') if f.strip()]
    # parts = line.strip().split(",")

    if len(parts) != NUM_SENSORS * 3:
        raise ValueError(f"Expected {NUM_SENSORS * 3} values from Teensy, got {len(parts)}: {line}")

    float_vals = [float(p) for p in parts]

    # Extract base values dynamically
    x, y, z = [], [], []
    for i in range(NUM_SENSORS):
        x.append(float_vals[i * 3])
        y.append(float_vals[i * 3 + 1])
        z.append(float_vals[i * 3 + 2])

    # Compute magnitudes and angles in XY plane
    mags = [np.sqrt(x[i] ** 2 + y[i] ** 2) for i in range(NUM_SENSORS)]
    angles_xy = [np.arctan2(y[i], x[i]) for i in range(NUM_SENSORS)]
    sins = [np.sin(a) for a in angles_xy]
    coss = [np.cos(a) for a in angles_xy]

    features = []

    # --- Standard Features ---
    if feature_xy:
        for i in range(NUM_SENSORS):
            features.extend([x[i], y[i]])

    if feature_z:
        for i in range(NUM_SENSORS):
            features.extend([z[i]])

    if feature_magnitude:
        features.extend(mags)

    if feature_angle:
        for i in range(NUM_SENSORS):
            features.extend([sins[i], coss[i]])

    # --- Engineered Relational Features ---
    # Only compute differences and products if there is more than 1 sensor
    if NUM_SENSORS >= 2:
        # Define the sensor pairs to compare based on the total count
        pairs = []
        if NUM_SENSORS == 2:
            pairs = [(0, 1)]  # Sensor 1 vs 2
        elif NUM_SENSORS == 3:
            pairs = [(0, 1), (1, 2), (0, 2)]  # 1 vs 2, 2 vs 3, 1 vs 3

        if feature_diff_xy:
            for i, j in pairs:
                features.extend([x[j] - x[i], y[j] - y[i]])

        if feature_diff_mag:
            for i, j in pairs:
                features.extend([mags[j] - mags[i]])

        if feature_dot_cross:
            for i, j in pairs:
                features.extend([
                    x[i] * x[j] + y[i] * y[j],  # Dot product
                    x[i] * y[j] - y[i] * x[j]  # Cross product
                ])

    return np.array([features], dtype=np.float32)

def build_header():
    # write header
    header = ["angle_joint"]
    if use_classic:
        for i in range(number_of_sensors):
            header.append(f"Sensor {i+1}")
    if use_diff: header.append("Sensor Diff")
    if use_nn: header.append("NN")
    if use_nn2: header.append("NN2")
    if use_poly: header.append("Poly")
    if use_poly2: header.append("Poly2")
    if use_poly3: header.append("Poly3")
    if use_poly4: header.append("Poly4")

    return header
def classic_prediction(line:str):
    # read line and calculate additonal features
    # fields = line.split(',')
    fields = [f.strip() for f in line.split(',') if f.strip()]
    values = [float(f) for f in fields]
    sensor_data = []

    for i in range(NUM_SENSORS):
        x = values[3 * i]
        y = values[3 * i + 1]

        magnitude = math.sqrt(x ** 2 + y ** 2)
        angle_deg = math.degrees(math.atan2(y, x))
        if i == 1 and shift_angle_2:
            angle_deg = angle_deg % 360 - 180  # ugly fix

        sensor_data.append((x, y, magnitude, angle_deg))
    # predict with both sensors
    predictions = []
    for i in range(NUM_SENSORS):
        x_measured_poly = poly.transform(np.array([[sensor_data[i][3]]]))
        predicted_joint_angle = models[i].predict(x_measured_poly)[0]
        predictions.append(predicted_joint_angle)

    return predictions


def diff_prediction(line:str):
    # read line and calculate additonal features
    #fields = line.split(',')
    fields = [f.strip() for f in line.split(',') if f.strip()]
    values = [float(f) for f in fields]
    sensor_data = []

    for i in range(number_of_sensors):
        x = values[3 * i]
        y = values[3 * i + 1]

        magnitude = math.sqrt(x ** 2 + y ** 2)
        angle_deg = math.degrees(math.atan2(y, x))
        if i == 1 and shift_angle_2:
            angle_deg = angle_deg % 360 - 180  # ugly fix

        sensor_data.append((x, y, magnitude, angle_deg))
    # predict with both sensors
    diff_x = sensor_data[0][0]- sensor_data[1][0]
    diff_y = sensor_data[0][1]- sensor_data[1][1]

    diff_angle = np.degrees(np.atan2(diff_y, diff_x)).reshape(-1,1)
    diff_poly = poly.fit_transform(diff_angle)

    predicted_joint_angle = model_diff.predict(diff_poly)[0]

    return predicted_joint_angle


def poly_prediction(line: str, num_sensors: int = NUM_SENSORS):
    # Read line and calculate additional features (corrected typo in your comment: "additional")
    # fields = line.split(',')
    fields = [f.strip() for f in line.split(',') if f.strip()]

    if not fields or fields[0] == '':
        return None

    values = [float(f) for f in fields]

    # Ensure the stream matches the expected sensor count
    if len(values) != num_sensors * 3:
        raise ValueError(f"Expected {num_sensors * 3} values, got {len(values)}")

    features = []

    # Dynamically extract and transform features for each sensor
    for i in range(num_sensors):
        x = values[i * 3]
        y = values[i * 3 + 1]
        z = values[i * 3 + 2]  # z is extracted but ignored in the final features, matching training

        # Calculate magnitude in the XY plane
        mag = np.sqrt(x ** 2 + y ** 2)

        # Calculate angle (np.arctan2 natively returns radians)
        angle_rad = np.arctan2(y, x)
        angle_deg = np.rad2deg(np.arctan2(y, x))

        # Calculate sine and cosine
        #sin_angle = np.sin(angle_rad)
        #cos_angle = np.cos(angle_rad)

        # Append exactly in the order the model was trained on: [sin, cos, mag]
        #features.extend([sin_angle, cos_angle, mag])
        features.extend([angle_deg])

    # Convert to a 2D array of shape (1, num_features) for scikit-learn
    X = np.array([features])

    # Predict using the polynomial pipeline
    predicted_angle = poly_model.predict(X)

    return predicted_angle[0]

def poly_prediction2(line: str, num_sensors: int = NUM_SENSORS):
    # Read line and calculate additional features (corrected typo in your comment: "additional")
    # fields = line.split(',')
    fields = [f.strip() for f in line.split(',') if f.strip()]

    if not fields or fields[0] == '':
        return None

    values = [float(f) for f in fields]

    # Ensure the stream matches the expected sensor count
    if len(values) != num_sensors * 3:
        raise ValueError(f"Expected {num_sensors * 3} values, got {len(values)}")

    features = []

    # Dynamically extract and transform features for each sensor
    for i in range(num_sensors):
        x = values[i * 3]
        y = values[i * 3 + 1]
        z = values[i * 3 + 2]  # z is extracted but ignored in the final features, matching training

        angle_deg = np.rad2deg(np.arctan2(y, x))

        features.extend([angle_deg])

    # Convert to a 2D array of shape (1, num_features) for scikit-learn
    X = np.array([features])

    # Predict using the polynomial pipeline
    predicted_angle = poly_model2.predict(X)

    return predicted_angle[0]

def poly_prediction3(line: str, num_sensors: int = NUM_SENSORS):
    # Read line and calculate additional features (corrected typo in your comment: "additional")
    # fields = line.split(',')
    fields = [f.strip() for f in line.split(',') if f.strip()]

    if not fields or fields[0] == '':
        return None

    values = [float(f) for f in fields]

    # Ensure the stream matches the expected sensor count
    if len(values) != num_sensors * 3:
        raise ValueError(f"Expected {num_sensors * 3} values, got {len(values)}")

    features = []

    # Dynamically extract and transform features for only the first two sensors
    for i in range(num_sensors-1):
        x = values[i * 3]
        y = values[i * 3 + 1]
        z = values[i * 3 + 2]  # z is extracted but ignored in the final features, matching training

        angle_deg = np.rad2deg(np.arctan2(y, x))

        features.extend([angle_deg])

    # Convert to a 2D array of shape (1, num_features) for scikit-learn
    X = np.array([features])

    # Predict using the polynomial pipeline
    predicted_angle = poly_model3.predict(X)

    return predicted_angle[0]

def poly_prediction4(line: str, num_sensors: int = NUM_SENSORS):
    # Read line and calculate additional features (corrected typo in your comment: "additional")
    # fields = line.split(',')
    fields = [f.strip() for f in line.split(',') if f.strip()]

    if not fields or fields[0] == '':
        return None

    values = [float(f) for f in fields]

    # Ensure the stream matches the expected sensor count
    if len(values) != num_sensors * 3:
        raise ValueError(f"Expected {num_sensors * 3} values, got {len(values)}")

    features = []

    # Dynamically extract and transform features for only the first two sensors
    for i in range(num_sensors-1):
        x = values[i * 3]
        y = values[i * 3 + 1]
        z = values[i * 3 + 2]  # z is extracted but ignored in the final features, matching training

        angle_deg = np.rad2deg(np.arctan2(y, x))

        features.extend([angle_deg])

    # Convert to a 2D array of shape (1, num_features) for scikit-learn
    X = np.array([features])

    # Predict using the polynomial pipeline
    predicted_angle = poly_model4.predict(X)

    return predicted_angle[0]


# -----------------------------
# Main loop
# -----------------------------
def main():
    global log_requested
    if use_classic:
        data = pd.read_csv(CLASSIC_DATA_PATH)
        x = []
        x_poly = []

        y = data['angle_joint'].values
        for i in range(number_of_sensors):
            x_i = data[f"angle_field_{i + 1}"].values.reshape(-1, 1)
            x_poly_i = poly.fit_transform(x_i)

            model_i = LinearRegression()
            model_i.fit(x_poly_i, y)

            x.append(x_i)
            x_poly.append(x_poly_i)
            models.append(model_i)

    if use_diff:
        global model_diff
        data = pd.read_csv(CLASSIC_DATA_PATH)
        x = []
        y_sens = []


        y = data['angle_joint'].values
        for i in range(NUM_SENSORS):
            x_i = data[f"x_{i + 1}"].values.reshape(-1, 1)
            y_i = data[f"y_{i + 1}"].values.reshape(-1, 1)
            x.append(x_i)
            y_sens.append(y_i)
        diff_x = x[0]- x[1]
        diff_y = y_sens[0]- y_sens[1]
        diff_angle = np.degrees(np.atan2(diff_y,diff_x))
        diff_poly = poly.fit_transform(diff_angle)

        model_diff = LinearRegression()
        model_diff.fit(diff_poly, y)

    if use_nn:
        model, scaler = load_model_and_scaler(1)
    if use_nn2:
        model2, scaler2 = load_model_and_scaler(2)

    if use_poly:
        global poly_model
        data = pd.read_csv(CLASSIC_DATA_PATH)

        # Target variable
        y = data['angle_joint'].values

        features = []

        # Dynamically extract and transform features for 1 to NUM_SENSORS
        for i in range(1, NUM_SENSORS + 1):
            mag_col = f'magnitude_{i}'
            angle_col = f'angle_field_{i}'

            if mag_col not in data.columns or angle_col not in data.columns:
                raise ValueError(f"Missing columns for sensor {i} in the dataset.")

            mag = data[mag_col].values
            angle_deg = data[angle_col].values

            # Calculate sine and cosine of the XY field angle
            angle_rad = np.deg2rad(angle_deg)
            sin_angle = np.sin(angle_rad)
            cos_angle = np.cos(angle_rad)

            # Append sin, cos, and magnitude to our feature list
            #features.extend([sin_angle, cos_angle, mag])
            features.extend([angle_deg])

        # Stack the list of arrays column-wise to create the final X matrix
        X = np.column_stack(features)
        # Set the polynomial degree (degree=2 creates x^2, y^2, and x*y interaction terms)
        poly_degree = 2

        # Create a pipeline that automatically expands features then fits the regression
        poly_model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=poly_degree, include_bias=False),
            LinearRegression()
        )

        # Fit the model to your data
        poly_model.fit(X, y)

    if use_poly2:
        global poly_model2
        data = pd.read_csv(CLASSIC_DATA_PATH)

        # Target variable
        y = data['angle_joint'].values

        features = []

        # Dynamically extract and transform features for 1 to NUM_SENSORS
        for i in range(1, NUM_SENSORS + 1):
            angle_col = f'angle_field_{i}'

            angle_deg = data[angle_col].values
            features.extend([angle_deg])

        # Stack the list of arrays column-wise to create the final X matrix
        X = np.column_stack(features)
        # Set the polynomial degree (degree=2 creates x^2, y^2, and x*y interaction terms)
        poly_degree = 3

        # Create a pipeline that automatically expands features then fits the regression
        poly_model2 = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=poly_degree, include_bias=False),
            LinearRegression()
        )

        # Fit the model to your data
        poly_model2.fit(X, y)

    if use_poly3:
        global poly_model3
        data = pd.read_csv(CLASSIC_DATA_PATH)

        # Target variable
        y = data['angle_joint'].values

        features = []

        # Dynamically extract and transform features for 1 to NUM_SENSORS, last sensor not used
        for i in range(1, NUM_SENSORS):
            angle_col = f'angle_field_{i}'

            angle_deg = data[angle_col].values
            features.extend([angle_deg])

        # Stack the list of arrays column-wise to create the final X matrix
        X = np.column_stack(features)
        # Set the polynomial degree (degree=2 creates x^2, y^2, and x*y interaction terms)
        poly_degree = 2

        # Create a pipeline that automatically expands features then fits the regression
        poly_model3 = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=poly_degree, include_bias=False),
            LinearRegression()
        )

        # Fit the model to your data
        poly_model3.fit(X, y)

    if use_poly4:
        global poly_model4
        data = pd.read_csv(CLASSIC_DATA_PATH)

        # Target variable
        y = data['angle_joint'].values

        features = []

        # Dynamically extract and transform features for 1 to NUM_SENSORS, last sensor not used
        for i in range(1, NUM_SENSORS):
            angle_col = f'angle_field_{i}'

            angle_deg = data[angle_col].values
            features.extend([angle_deg])

        # Stack the list of arrays column-wise to create the final X matrix
        X = np.column_stack(features)
        # Set the polynomial degree (degree=2 creates x^2, y^2, and x*y interaction terms)
        poly_degree = 3

        # Create a pipeline that automatically expands features then fits the regression
        poly_model4 = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=poly_degree, include_bias=False),
            LinearRegression()
        )

        # Fit the model to your data
        poly_model4.fit(X, y)

    if logging_enabled:
        # Listen for 'L' (non-blocking)
        keyboard.on_press_key('l', request_log_entry)
        try:
            with open(LOG_FILE, 'x', newline='') as f:
                csv.writer(f).writerow(build_header())
        except FileExistsError:
            pass

    partial_line = b""
    latest_line = None

    print("Opening serial port...")
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)


    print("\n--- LIVE INFERENCE STARTED ---")
    print("Reading data from sensors and predicting angle...\n")
    if plot_output:
        plotter = LivePlotter(window=70)
    while True:
        try:
            while True:
                # Read serial data
                data = ser.read(ser.in_waiting or 1)
                if data:
                    partial_line += data
                    lines = partial_line.split(b'\n')
                    partial_line = lines[-1]
                    for raw in lines[:-1]:
                        line = raw.decode('utf-8', errors='ignore').strip()
                        if line:
                            latest_line = line

                    # Parse raw sensor data
                    X = parse_teensy_line(latest_line)
                    all_predictions = []
                    if use_classic:
                        predictions = classic_prediction(latest_line)
                        for i, pred in enumerate(predictions, start=1):
                            all_predictions.append({
                                "type": f"Sensor {i}",
                                "value": round(pred,3)
                            })

                    if use_diff:
                        prediction_diff = diff_prediction(latest_line)
                        all_predictions.append({
                            "type": f"Sensor Diff",
                            "value": round(prediction_diff,3)
                            })
                    if use_nn:
                        # Scale
                        X_scaled = scaler.transform(X)
                        # Predict
                        angle, uncertainty = predict_with_uncertainty(
                            model, X_scaled, mc_samples=MC_DROPOUT_SAMPLES
                        )

                        all_predictions.append({
                            "type": "NN",
                            "value": round(angle,3)
                        })
                        if MC_DROPOUT_SAMPLES:
                            all_predictions.append({
                                "type": "NN_uncertainty",
                                "value": uncertainty
                            })

                    if use_nn2:
                        # Scale
                        if second_net_double_sensor:
                            # Hardcoded indices to keep for NUM_SENSORS=2 when all flags are True
                            X2 = X[:, [0, 1, 2, 3, 6, 7, 9, 10, 12, 13, 14, 15, 18, 19, 24, 27, 28]]
                        else:
                            X2 = X
                        X_scaled = scaler2.transform(X2)
                        # Predict
                        angle, uncertainty = predict_with_uncertainty(
                            model2, X_scaled, mc_samples=MC_DROPOUT_SAMPLES
                        )

                        all_predictions.append({
                            "type": "NN2",
                            "value": round(angle,3)
                        })
                        if MC_DROPOUT_SAMPLES:
                            all_predictions.append({
                                "type": "NN2_uncertainty",
                                "value": uncertainty
                            })

                    if use_poly:
                        prediction_poly = poly_prediction(latest_line)
                        all_predictions.append({
                            "type": f"Poly",
                            "value": round(prediction_poly, 3)
                        })
                    if use_poly2:
                        prediction_poly2 = poly_prediction2(latest_line)
                        all_predictions.append({
                            "type": f"Poly2",
                            "value": round(prediction_poly2, 3)
                        })
                    if use_poly3:
                        prediction_poly3 = poly_prediction3(latest_line)
                        all_predictions.append({
                            "type": f"Poly3",
                            "value": round(prediction_poly3, 3)
                        })
                    if use_poly4:
                        prediction_poly4 = poly_prediction4(latest_line)
                        all_predictions.append({
                            "type": f"Poly4",
                            "value": round(prediction_poly4, 3)
                        })

                    print(", ".join([f"{d['type']}: {d['value']:6.2f}" for d in all_predictions]))


                    if log_requested:
                        print("Logging data point")  # Optional feedback

                        # 1. Gather data
                        row = [joint_angle]
                        for d in all_predictions:
                            row.append(d['value'])

                        # 2. Write to file (Append mode)
                        with open(LOG_FILE, 'a', newline='') as f:
                            csv.writer(f).writerow(row)
                        # 3. Reset the flag so we don't log again until L is pressed
                        log_requested = False

                    if plot_output:
                        plotter.update(all_predictions)

                    time.sleep(intervall/1000)

        except ValueError as e:
            print(f"Parse error: {e}")

        except KeyboardInterrupt:
            print("Stopping inference.")
            break

        except Exception as ex:
            print("Error:", ex)


if __name__ == "__main__":
    main()
