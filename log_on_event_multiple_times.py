import math
import serial
import time
import csv
import keyboard
import numpy as np
"""
logs a burst of data when pressing L
calculates magnitude and angle in xy-plane and also logs this
"""


# --- CONFIG ---
PORT = 'COM7'
BAUD = 115200
FILENAME = 'logs/PCB_v2/Experiment 3/mcp_3.csv'
number_of_sensors = 3
xyz_mode = True     # <--- ENABLE XYZ MODE
logs_in_burst = 1  # logs per angle
time_between_logs = 10  # number of ms between each log

angle_step = 2
min_angle = 72
max_angle = 72
# --------------

ser = serial.Serial(PORT, BAUD, timeout=0)
time.sleep(2)

partial_line = b""
latest_line = None
header_written = False
counter = 0
logging_enabled = True
logging_burst_ongoing = False
burst_counter = 0

# field count per sensor
fields_per_sensor = 3 if xyz_mode else 2

angles = np.arange(min_angle, max_angle + angle_step, angle_step)


def log_latest_line():
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    count = (counter-1)*logs_in_burst+burst_counter+1

    try:
        fields = [f.strip() for f in latest_line.split(',') if f.strip()]
        if len(fields) != number_of_sensors * fields_per_sensor:
            raise ValueError("Wrong number of fields for sensors")
        # Round the initial float conversion to 3 decimal places
        values = [round(float(f), 3) for f in fields]

    except ValueError:
        print(f"Skipping invalid data: {latest_line}")
        return False

    # -------- Parse sensor data --------
    sensor_data = []

    for i in range(number_of_sensors):
        base = i * fields_per_sensor
        x = values[base]
        y = values[base + 1]
        z = values[base + 2] if xyz_mode else None

        # Round derived calculations to 3 decimal places
        magnitude = round(math.sqrt(x * x + y * y), 3)
        angle_deg = round(math.degrees(math.atan2(y, x)), 3)

        sensor_data.append((x, y, z, magnitude, angle_deg))

    # -------- Build data row --------
    row = [angles[counter - 1], count, t_str]

    for (x, y, z, mag, ang) in sensor_data:
        row += [x, y]
        if xyz_mode:
            row.append(z)
        row += [mag, ang]

    writer.writerow(row)

    print(f"Logged #{count} | Sensors: {sensor_data}")
    return True


def build_header():
    # write header
    header = ["angle_joint", "counter", "timestamp"]

    for i in range(number_of_sensors):
        header += [f"x_{i + 1}", f"y_{i + 1}"]
        if xyz_mode:
            header.append(f"z_{i + 1}")
        header += [
            f"magnitude_{i + 1}",
            f"angle_field_{i + 1}"
        ]
    return header


with open(FILENAME, 'w', newline='', buffering=1) as csvfile:
    writer = csv.writer(csvfile)
    print(f"Press 'L' to start logging burst of {logs_in_burst} logs with {time_between_logs}ms in between")
    print(f"Please move joint to angle: {angles[0]}°")

    writer.writerow(build_header())

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

            # automatically log if burst is ongoing
            if latest_line and logging_burst_ongoing:
                if burst_counter > logs_in_burst-1:
                    logging_burst_ongoing = False
                    burst_counter = 0
                    if counter == len(angles):
                        print("\nLogging completed.")
                        ser.close()
                        exit()

                    print(f"Please move joint to angle: {angles[counter]}°")
                    logging_enabled = False
                else:
                    log_latest_line()
                    burst_counter += 1
                    time.sleep(time_between_logs*0.001)


            # Logging started on 'L'
            elif keyboard.is_pressed('l') and logging_enabled:
                counter += 1
                logging_burst_ongoing = True



            # Reset trigger
            if not keyboard.is_pressed('l'):
                logging_enabled = True

    except KeyboardInterrupt:
        print("\nLogging stopped.")
        ser.close()
