// Arduino Nano 33 BLE
// by Nikola Georgiev

#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <Arduino_LPS22HB.h>
#include <Arduino_HTS221.h>

char name[] = "Red";
byte pollingRate = 0;

unsigned long pollTime;
unsigned long newCount;
unsigned long pollCount;
unsigned long maxCount;

BLEService logService("eb03300a-160b-4c02-91d7-75cef930279f");
BLECharacteristic allData("9cfd98a5-7bb9-4ae9-b0de-8a81ee6800c9", BLENotify, 40);
BLEByteCharacteristic setPollRate("553f87f5-351a-49ed-83a4-2a640c0c7084", BLEWrite);

float x_mag = 0.0;
float y_mag = 0.0;
float z_mag = 0.0;
float x_acc = 0.0;
float y_acc = 0.0;
float z_acc = 0.0;
float x_gyro = 0.0;
float y_gyro = 0.0;
float z_gyro = 0.0;

byte packet[40] = {};

void resetCounters()
{
  pollTime = 0;
  newCount = 0;
  pollCount = 0;
  // record the timestep count where micros() overflows (max of unsigned long)
  maxCount = 4294967295 / ((float)1000000 / pollingRate);
}

// WARNING: micros() will overflow after approx 72 min
void updateCounters()
{
  pollTime = micros();
  newCount = (float)pollTime / ((float)1000000 / pollingRate);
}

void updatePollRate()
{
  if (setPollRate.written())
  {
    pollingRate = setPollRate.value();
    Serial.println("Poll Rate Set:");
    Serial.println(pollingRate);
    if (pollingRate <= 0)
    {
      return;
      // must have positive sample rate to send data
    }
    else
    {
      // changing the poll rate affects the update count, so reset
      resetCounters();
    }
  }
}

void setup()
{
  Serial.begin(9600);
  // Will just start straightaway since designed to work over battery
  if (!IMU.begin())
  {
    Serial.println("Failed to initialize IMU!");
    while (1)
    {
    }
  }
  else if (!BLE.begin())
  {
    Serial.println("Starting BLE failed!");
    while (1)
    {
    }
  }

  BLE.setDeviceName(name);
  BLE.setLocalName(name);

  logService.addCharacteristic(allData);
  logService.addCharacteristic(setPollRate);

  BLE.setAdvertisedService(logService);

  BLE.addService(logService);

  allData.writeValue(packet, 40);
  setPollRate.writeValue(0);

  BLE.advertise();

  Serial.println("Waiting for connections...");
}

void assemblePacket()
{
  byte tmp[4];
  *((float *)tmp) = x_acc;
  packet[0] = tmp[0];
  packet[1] = tmp[1];
  packet[2] = tmp[2];
  packet[3] = tmp[3];

  *((float *)tmp) = y_acc;
  packet[4] = tmp[0];
  packet[5] = tmp[1];
  packet[6] = tmp[2];
  packet[7] = tmp[3];

  *((float *)tmp) = z_acc;
  packet[8] = tmp[0];
  packet[9] = tmp[1];
  packet[10] = tmp[2];
  packet[11] = tmp[3];

  *((float *)tmp) = x_gyro;
  packet[12] = tmp[0];
  packet[13] = tmp[1];
  packet[14] = tmp[2];
  packet[15] = tmp[3];

  *((float *)tmp) = y_gyro;
  packet[16] = tmp[0];
  packet[17] = tmp[1];
  packet[18] = tmp[2];
  packet[19] = tmp[3];

  *((float *)tmp) = z_gyro;
  packet[20] = tmp[0];
  packet[21] = tmp[1];
  packet[22] = tmp[2];
  packet[23] = tmp[3];

  *((float *)tmp) = x_mag;
  packet[24] = tmp[0];
  packet[25] = tmp[1];
  packet[26] = tmp[2];
  packet[27] = tmp[3];

  *((float *)tmp) = y_mag;
  packet[28] = tmp[0];
  packet[29] = tmp[1];
  packet[30] = tmp[2];
  packet[31] = tmp[3];

  *((float *)tmp) = z_mag;
  packet[32] = tmp[0];
  packet[33] = tmp[1];
  packet[34] = tmp[2];
  packet[35] = tmp[3];

  *((unsigned long *)tmp) = pollTime;
  packet[36] = tmp[0];
  packet[37] = tmp[1];
  packet[38] = tmp[2];
  packet[39] = tmp[3];
}

void loop()
{
  BLEDevice central = BLE.central();

  if (central)
  {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    while (central.connected())
    {

      updatePollRate();

      if (pollingRate <= 0)
      {
        continue;
      }

      updateCounters();

      // send new data
      // when at least 1/pollingRate time has passed OR if micros() has overflowed
      if (newCount > pollCount || (pollCount - newCount) >= maxCount / 2)
      {

        IMU.readMagneticField(x_mag, y_mag, z_mag);
        IMU.readAcceleration(x_acc, y_acc, z_acc);
        IMU.readGyroscope(x_gyro, y_gyro, z_gyro);

        assemblePacket();
        allData.writeValue(packet, 40);

        pollCount = newCount;

        delay(((float)(newCount + 1) * ((float)1000000 / pollingRate) - micros()) / 1000);
        // delaying a millisecond amount leads to more consistent rates than delayMicroseconds (not sure why)
      }
    }

    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
  }
}
