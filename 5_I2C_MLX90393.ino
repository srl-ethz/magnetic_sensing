// #include "Fast_MLX90393.h" // can for now also be replaced with the standard Adafruit_MLX90393 since nothing much has changed
#include "Adafruit_MLX90393.h"

// Define the number of sensors to keep the code scalable
/*
// Initialize an array of sensor objects
#define NUM_SENSORS 5
Adafruit_MLX90393 sensors[NUM_SENSORS] = {
  Adafruit_MLX90393(),
  Adafruit_MLX90393(),
  Adafruit_MLX90393(),
  Adafruit_MLX90393(),
  Adafruit_MLX90393()
};

// MLX90393-011 addresses: 0x0C, 0x0D, 0x0E, 0x0F
// MLX90393-012 address (both pins grounded): 0x10
// sensors 1-3 are MCP, 4-5 are PIP
const uint8_t MLX_ADDR[NUM_SENSORS] = {0x0C, 0x0D, 0x0E, 0x0F, 0x10};
*/

// use only mcp
#define NUM_SENSORS 3

// Initialize an array of sensor objects

Adafruit_MLX90393 sensors[NUM_SENSORS] = {
  Adafruit_MLX90393(),
  Adafruit_MLX90393(),
  Adafruit_MLX90393()
};

// MLX90393-011 addresses: 0x0C, 0x0D, 0x0E, 0x0F
// MLX90393-012 address (both pins grounded): 0x10
// sensors 1-3 are MCP
const uint8_t MLX_ADDR[NUM_SENSORS] = {0x0C, 0x0D, 0x0E};
// use only pip
/*
#define NUM_SENSORS 2
Adafruit_MLX90393 sensors[NUM_SENSORS] = {
  Adafruit_MLX90393(),
  Adafruit_MLX90393()
};

// MLX90393-011 addresses: 0x0C, 0x0D, 0x0E, 0x0F
// MLX90393-012 address (both pins grounded): 0x10
// sensors 1-3 are MCP, 4-5 are PIP
// const uint8_t MLX_ADDR[NUM_SENSORS] = {0x0F, 0x10};
const uint8_t MLX_ADDR[NUM_SENSORS] = {0x0C, 0x0D};
*/


#define GAIN MLX90393_GAIN_3X
#define RES_X MLX90393_RES_18
#define RES_Y MLX90393_RES_18
#define RES_Z MLX90393_RES_18
#define OVERSAMPLING MLX90393_OSR_2
#define FILTER MLX90393_FILTER_1

bool xy_mode = false;
bool xyz_mode = true;
bool direction_mode = false;
bool diff_mode = false;
bool time_measurement = false;

float delay_measurement = 500; // 7: (for osr2, fil 1), 3: (for osr1, fil1)

#define SCL1_PIN 19
#define SDA1_PIN 18

void I2C_Bus_Recovery() {
  Serial.println("Attempting I2C Bus Recovery...");
  pinMode(SCL1_PIN, OUTPUT);
  pinMode(SDA1_PIN, INPUT);

  // Clock the SCL line 10 times to force sensors to release SDA
  for (int i = 0; i < 10; i++) {
    digitalWrite(SCL1_PIN, HIGH);
    delayMicroseconds(5);
    digitalWrite(SCL1_PIN, LOW);
    delayMicroseconds(5);
  }
  
  Serial.println("Bus Recovery Complete.");
}



void setup(void)
{
  Serial.begin(115200);

  /* Wait for serial on USB platforms. */
  while (!Serial) {
      delay(10);
  }

  I2C_Bus_Recovery();
  
  Serial.println("Starting 5x I2C MLX90393");
  Wire.setClock(100000);
  bool success = true;
  // Loop through all sensors to reset, begin, and configure them
  for (int i = 0; i < NUM_SENSORS; i++) {

    // 1. Try to begin
    if (!sensors[i].begin_I2C(MLX_ADDR[i], &Wire)) { 
      Serial.print("Sensor "); Serial.print(i + 1);
      Serial.print(" (0x"); Serial.print(MLX_ADDR[i], HEX);
      Serial.println(") NOT FOUND");
      success = false;
      delay(1000);
      continue; // SKIP the rest of the config for this sensor!
    }

    // 2. Only configure if found
    Serial.print("Found! ... Configuring Sensor "); Serial.println(i + 1);
    
    //sensors[i].reset(); 
    //delay(50); // Give the chip time to wake up after reset

    if (!sensors[i].setGain(GAIN)) Serial.println("Gain failed");
    sensors[i].setResolution(MLX90393_X, RES_X);
    sensors[i].setResolution(MLX90393_Y, RES_Y);
    sensors[i].setResolution(MLX90393_Z, RES_Z);
    sensors[i].setOversampling(OVERSAMPLING);
    sensors[i].setFilter(FILTER);
    
    sensors[i].startSingleMeasurement();
    delay(20); 
  }
  if (!success){
    Serial.println("No success starting communication ... rebooting");
    delay(4000);
    _reboot_Teensyduino_();
  }
  
  delay(10);
}

void loop(void) {
  // Arrays to hold data for all 5 sensors
  float x[NUM_SENSORS], y[NUM_SENSORS], z[NUM_SENSORS];
  delay(delay_measurement);

  // Read data from all sensors
  for (int i = 0; i < NUM_SENSORS; i++) {
    if (sensors[i].readMeasurement(&x[i], &y[i], &z[i])) {
      sensors[i].startSingleMeasurement();
      
      if (direction_mode) {
        double direction = atan2(y[i], x[i]) * (180.0 / PI);
        Serial.print(direction, 4);
        Serial.print(", ");
      }
      if (xy_mode) {
        Serial.print(x[i], 4);
        Serial.print(", ");
        Serial.print(y[i], 4);
        Serial.print(", ");
      }
      if (xyz_mode) {
        Serial.print(x[i], 4);
        Serial.print(", ");
        Serial.print(y[i], 4);
        Serial.print(", ");
        Serial.print(z[i], 4);
        Serial.print(", ");
      }
    } else {
      Serial.print("Unable to read XYZ data from sensor ");
      Serial.println(i + 1);
      sensors[i].startSingleMeasurement();
      delay(delay_measurement * 10);
    }
  }

  // Differential mode: calculates the difference between Sensor 1 (reference) and Sensors 2-5
  if (diff_mode) {
    for (int i = 1; i < NUM_SENSORS; i++) {
      float x_diff = x[0] - x[i];
      float y_diff = y[0] - y[i];
      float z_diff = z[0] - z[i];
      
      Serial.print(x_diff, 4);
      Serial.print(", ");
      Serial.print(y_diff, 4);
      Serial.print(", ");
      Serial.print(z_diff, 4);
      Serial.print(", ");
    }
  }
  
  // Terminate the row of data for the Serial Plotter/Monitor
  Serial.println();

  if (time_measurement) {
    static uint32_t last_us = 0;    
    static uint32_t sum_dt = 0;
    static uint32_t count = 0;

    uint32_t now = micros();

    if (last_us != 0) {
        sum_dt += (now - last_us);
        count++;

        if (count == 200) {
            float avg_dt_us = sum_dt / (float)count;
            float hz = 1e6 / avg_dt_us;

            Serial.print("Effective rate: ");
            Serial.print(hz, 1);
            Serial.println(" Hz");

            sum_dt = 0;
            count = 0;
        }
    }
    last_us = now;
  }
}