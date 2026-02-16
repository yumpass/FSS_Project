#include <AutoPID.h>
#include "InterpolationLib.h"

#define calefactor 9
#define OUTPUT_MIN 0
#define OUTPUT_MAX 255
#define KP 70
#define KI 1
#define KD 80


String input;
int adc_in;
double t_[64] = { 300, 295, 290, 285, 280, 275, 270, 265, 260, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15 };
double adc_[64] = { 23, 25, 27, 28, 31, 33, 35, 38, 41, 44, 48, 52, 56, 61, 66, 71, 78, 84, 92, 100, 109, 120, 131, 143, 156, 171, 187, 205, 224, 245, 268, 293, 320, 348, 379, 411, 445, 480, 516, 553, 591, 628, 665, 702, 737, 770, 801, 830, 857, 881, 903, 922, 939, 954, 966, 977, 985, 993, 999, 1004, 1008, 1012, 1016, 1020 };
double temperature, setPoint = 0, outputVal;

AutoPID myPID(&temperature, &setPoint, &outputVal, OUTPUT_MIN, OUTPUT_MAX, KP, KI, KD);


void set() {
  
  if (Serial.available() > 0) {
    input = Serial.readStringUntil('\n');
    input.trim();
    double sp = input.toFloat();
    setPoint = sp;
    Serial.print("SP=");
    Serial.println(setPoint, 2);
  }
  

}

void setup() {
  Serial.begin(9600);
  pinMode(calefactor, OUTPUT);


  myPID.setBangBang(20);
  myPID.setTimeStep(1);
}


void loop() {
  adc_in = analogRead(A14);
  temperature = Interpolation::Linear(adc_, t_, 64, adc_in, false);
  set();

  myPID.run();

  analogWrite(calefactor, outputVal);

  static unsigned long last = 0;
if (millis() - last >= 200) {
  last = millis();
  Serial.print("T=");
  Serial.println(temperature, 2);
}

}
