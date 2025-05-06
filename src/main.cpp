#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include <signalProcessing.h>

// Khai báo đối tượng MAX30105 (có thể dùng cho MAX30102)
MAX30105 particleSensor;

int32_t ir_avg = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin(); // Khởi tạo I2C (nếu dùng chân khác thì sửa Wire.begin(SDA, SCL);)

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 was not found. Please check wiring/power.");
    while (1);
  }

  // Cấu hình cảm biến MAX30102  
  byte ledBrightness = 0x1F; // Độ sáng LED (từ 0x00 đến 0xFF)
  byte sampleAverage = 8;   // Trung bình mẫu (1, 2, 4, 8, 16, 32)
  byte ledMode = 2;         // 2 LED mode (Red + IR)
  int sampleRate = 400;     // Tần số lấy mẫu (50, 100, 200, 400, 800, 1000, 1600, 3200)
  int pulseWidth = 411;     // Độ rộng xung (69, 118, 215, 411) (đơn vị us)
  int adcRange = 4096;     // Dải ADC (2048, 4096, 8192, 16384)

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  Serial.println("MAX30102 ready to read PPG...");

  const byte avgAmount = 64;
  long baseValue = 0;
  for (byte x = 0 ; x < avgAmount ; x++)
  {
    baseValue += particleSensor.getIR(); //Read the IR value
  }
  baseValue /= avgAmount;

  //Pre-populate the plotter so that the Y scale is close to IR values
  for (int x = 0 ; x < 500 ; x++)
    Serial.println(baseValue);
}

void loop() {


  long irValue = particleSensor.getIR();
  Serial.print(irValue);
  int16_t filteredDC = DCfilter(&ir_avg, irValue);
  int16_t filteredFIR = irValue - filteredDC;
  
  Serial.print(",");
  Serial.println(filteredFIR);
}
