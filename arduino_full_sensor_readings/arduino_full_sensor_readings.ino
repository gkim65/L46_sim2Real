/*
  US-100 Ultrasonic Sensor - Serial Mode
  modified on 26 Sep 2020
  by Mohammad Reza Akbari @ Electropeak
  Home
*/

// DISTANCE SENSORS --------------------------------------
#include <SoftwareSerial.h>

SoftwareSerial mySerial(3, 2);
SoftwareSerial my2Serial(4, 5);

// Reading in for ultrasonic distance sensor
unsigned int HighByte = 0;
unsigned int LowByte  = 0;
double Len1  = 0;
double Len1_2  = 0;
double Len2  = 0;
double Len2_2  = 0;
double velocity1  = 0;
double velocity2  = 0;
double distance_1  = 0;
double distance_2  = 0;

// IMU ANGLES --------------------------------------
#include<Wire.h>
const int MPU_addr=0x68;
int16_t AcX,AcY,AcZ,Tmp,GyX,GyY,GyZ;

int minVal=265;
int maxVal=402;
 
double x;
double y;
double z;

// Accelerometers -------------------------------
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;
#define PI 3.1415926535897932384626433832795





void setup(void) {
  Serial.begin(9600);
  mySerial.begin(9600);
  my2Serial.begin(9600);

  Wire.begin();
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit MPU6050 test!");

  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }

  Serial.println("");
  delay(100);
}

void loop() {
  mySerial.listen();
  mySerial.flush();
  mySerial.write(0X55);                          // trig US-100 begin to measure the distance
  delay(50);                                  
  if (mySerial.available() >= 2)                  // check receive 2 bytes correctly
  {
    HighByte = mySerial.read();
    LowByte  = mySerial.read();
    Len1  = (double) HighByte * 256 + LowByte;          // Calculate the distance
    mySerial.listen();
    mySerial.flush();
    mySerial.write(0X55);                          // trig US-100 begin to measure the distance
    delay(50);  
    HighByte = mySerial.read();
    LowByte  = mySerial.read();
    Len1_2  = (double) HighByte * 256 + LowByte;          // Calculate the distance
    velocity1 = (Len1_2-Len1)/50;
  }
  my2Serial.listen();
  my2Serial.flush();
  my2Serial.write(0X55);                          // trig US-100 begin to measure the distance
  delay(50);                                  
  if (my2Serial.available() >= 2)                  // check receive 2 bytes correctly
  {
    HighByte = my2Serial.read();
    LowByte  = my2Serial.read();
    Len2  = (double) HighByte * 256 + LowByte;          // Calculate the distance
    my2Serial.listen();
    my2Serial.flush();
    my2Serial.write(0X55);                          // trig US-100 begin to measure the distance
    delay(50);  
    HighByte = my2Serial.read();
    LowByte  = my2Serial.read();
    Len2_2  = (double) HighByte * 256 + LowByte;          // Calculate the distance
    velocity2 = (Len2_2-Len2)/50;
  }

  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_addr,14,true);
  AcX=Wire.read()<<8|Wire.read();
  AcY=Wire.read()<<8|Wire.read();
  AcZ=Wire.read()<<8|Wire.read();
  int xAng = map(AcX,minVal,maxVal,-90,90);
  int yAng = map(AcY,minVal,maxVal,-90,90);
  int zAng = map(AcZ,minVal,maxVal,-90,90);
  
  x= RAD_TO_DEG * (atan2(-yAng, -zAng)+PI);
  y= RAD_TO_DEG * (atan2(-xAng, -zAng)+PI);
  z= RAD_TO_DEG * (atan2(-yAng, -xAng)+PI);

  /* Get new sensor events with the readings */
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  distance_1 = (Len1_2 - 370)/1000;
  distance_2 = (370 - Len2_2)/1000;


  if ((Len1 > 1) && (Len1 < 10000) && (Len2 > 1) && (Len2 < 10000))
    {
      Serial.println("Distance in (mm): ");
      Serial.print(Len1);
      Serial.print(" , "); 
      Serial.print(Len1_2);
      Serial.print(" , "); 
      Serial.print(distance_1);
      Serial.print(" , "); 
      Serial.print(velocity1);
      Serial.print(" , "); 
      Serial.print(Len2);    
      Serial.print(" , ");
      Serial.print(Len2_2);
      Serial.print(" , ");
      Serial.print(distance_2);
      Serial.print(" , "); 
      Serial.print(velocity2);
      Serial.print(" , ");  
      Serial.print(x);
      Serial.print(" , ");
      Serial.println(g.gyro.x*180/PI);

    }
  
  delay(100);                                    
} 
