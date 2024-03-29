import time
import serial
import adafruit_us100
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=.1)
us100 = adafruit_us100.US100(uart)
while True:
    print("-----")
    print("Temperature: ", us100.temperature)
    print("Distance: ", us100.distance)
    time.sleep(0.5)