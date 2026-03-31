# magnetic_sensing
Use magnetic sensing to predict the joint angle on a rolling contact joint based robotic hand (faive)

## 🛠️ Project Structure

### 🐍 Python Scripts
| File | Description |
| :--- | :--- |
| `find_optimal_pymoo.py` | Multi-objective optimization using PyMoo and NSGA-II. |
| `find_optimal_scipy.py` | Single-objective optimization using SciPy. |
| `visualisation3.py` | 2D visualization of magnetic fields. |
| `triple_sensor.py` | Plot generation for three sensor setup. |
| `test_config_robustness_running.py` | Robustness testing. |
| `generate_training_data.py` | Synthetic data generation for training of neural net. |
| `helper_functions.py` | Core utility functions. |
| `helper_plotting.py` | Plotting utility functions. |
| `showcase_pareto_front.py` | Plots the Pareto Front. |

#### 🤖 Prediction and Logging
* `log_on_event_multiple_times.py`: Data logging utility.
* `train_model_v2.py`: Model training (v2).
* `inference_v2.py`: Real-time inference (v2).

### 🤖 Arduino Scripts
* `5_I2C_MLX90393.ino`: MLX90393 I2C sensor firmware.


