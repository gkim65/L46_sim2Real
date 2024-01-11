import time
import serial
import adafruit_us100
ser = serial.Serial("/dev/ttyACM0", baudrate=9600, timeout=.01)
ser.reset_input_buffer()

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
