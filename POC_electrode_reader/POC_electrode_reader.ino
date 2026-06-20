#include "esp_system.h"
#include <Arduino.h>
#include <math.h>

const int PIN_PLANT = 11;   // change if you used a different ADC pin

void setup() {
  Serial.begin(115200);
  while(!Serial){
    ;
  }
  analogReadResolution(12);     // ESP32-S3: 0–4095
  analogSetAttenuation(ADC_11db); // good for 0–3.3V range
  Serial.println("Start");
}

void loop() {
  int raw = analogRead(PIN_PLANT);   // 0–4095
  float volts = raw * (3.3f / 4095.0f);

  // Serial.print("RAW: ");
  // Serial.print(raw);
  // Serial.print("   VOLTS: ");
  Serial.print(volts, 4);
  Serial.println(",");

  delay(10);  // ~50 Hz sampling
}