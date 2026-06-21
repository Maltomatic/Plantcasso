#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// PWM pulse range for typical analog servos, in "ticks" at 50Hz
#define SERVO_MIN  102   // ~1ms pulse
#define SERVO_MAX  512   // ~2ms pulse
float STARTUP_POS[5] = {90.0f, 90.0f, 10.0f, 10.0f, 90.0f};

void setup() {
  Serial.begin(115200);
  while(!Serial);
  Wire.begin(1, 2); // SDA, SCL
  delay(1000);
  Serial.println("Scanning...");
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("Found device at 0x");
      Serial.println(addr, HEX);
    }
  }
  pwm.begin();
  pwm.sleep();
  delay(5);
  pwm.wakeup();
  pwm.setPWMFreq(50);     // standard servo refresh rate
}

void setAngle(uint8_t channel, float angleDeg) {
  uint16_t pulse = map(angleDeg, 0, 180, SERVO_MIN, SERVO_MAX);
  pwm.setPWM(channel, 0, pulse);
}
int inc = 5;
void loop() {
  for (uint8_t ch = 0; ch < 5; ch++) {
    Serial.printf("Set to "); Serial.println(STARTUP_POS[ch]);
    setAngle(ch, STARTUP_POS[ch]);
  }
  delay(100);
}