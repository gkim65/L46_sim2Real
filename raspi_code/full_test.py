import RPi.GPIO as GPIO
from time import sleep

# Setting up Motors
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
Motor1 = {'EN': 25, 'input1': 24, 'input2': 23}
for x in Motor1:
    GPIO.setup(Motor1[x], GPIO.OUT)

EN1 = GPIO.PWM(Motor1['EN'], 100)
EN1.start(0)

#!/usr/bin/python
# Just telling us which python we're going to use for the interpreter


import smbus
import math
import time

power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c


def read_byte(adr):
    return bus.read_byte_data(address, adr)


def read_word(adr):
    high = bus.read_byte_data(address, adr)
    low = bus.read_byte_data(address, adr + 1)
    val = (high << 8) + low
    return val


def read_word_2c(adr):
    val = read_word(adr)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val


def dist(a, b):
    return math.sqrt((a * a) + (b * b))


def get_y_rotation(x, y, z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)


def get_x_rotation(x, y, z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)


bus = smbus.SMBus(1)
address = 0x68

bus.write_byte_data(address, power_mgmt_1, 0)


import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import datetime as dt

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xs = []
ys = []

def animate(angle,xs,ys):
    accel_xout = read_word_2c(0x3b)
    accel_yout = read_word_2c(0x3d)
    accel_zout = read_word_2c(0x3f)

    accel_xout_scaled = accel_xout / 16384.0
    accel_yout_scaled = accel_yout / 16384.0
    accel_zout_scaled = accel_zout / 16384.0

    print("{}\t{}\t{}\t{}".format ("X out: ", accel_xout, " scaled: ", accel_xout_scaled)) 
    print("{}\t{}\t{}\t{}".format ("Y out: ", accel_yout, " scaled: ", accel_yout_scaled)) 
    print("{}\t{}\t{}\t{}".format ("Z out: ", accel_zout, " scaled: ", accel_zout_scaled)) 
	
    print()
    angle = get_x_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)

    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(angle)

    xs = xs[-20:]
    ys = ys[-20:]

    ax.clear()
    ax.plot(xs,ys)


import time
import serial
import adafruit_us100
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=.1)
us100 = adafruit_us100.US100(uart)

# Without Distance
while True:
    accel_xout = read_word_2c(0x3b)
    accel_yout = read_word_2c(0x3d)
    accel_zout = read_word_2c(0x3f)

    accel_xout_scaled = accel_xout / 16384.0
    accel_yout_scaled = accel_yout / 16384.0
    accel_zout_scaled = accel_zout / 16384.0
    angle = get_x_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled) 
    print(angle)
    if angle > -63:
        angle = -63
    if angle < -72:
        angle = -72

    if angle < -68:     
        EN1.ChangeDutyCycle(100)
        GPIO.output(Motor1['input1'], GPIO.HIGH)
        GPIO.output(Motor1['input2'], GPIO.LOW)
        sleep(0.05)
    else:     
        EN1.ChangeDutyCycle(100)
        GPIO.output(Motor1['input1'], GPIO.LOW)
        GPIO.output(Motor1['input2'], GPIO.HIGH)
        sleep(0.05)
        





# With Distance
# while True:
#     distance = us100.distance
#     print("Distance: ", distance)

#     if distance < 10:

#         EN1.ChangeDutyCycle(100)
#         GPIO.output(Motor1['input1'], GPIO.HIGH)
#         GPIO.output(Motor1['input2'], GPIO.LOW)
#         sleep(0.25)
#     elif distance > 60:

#         EN1.ChangeDutyCycle(100)
#         GPIO.output(Motor1['input1'], GPIO.LOW)
#         GPIO.output(Motor1['input2'], GPIO.HIGH)
#         sleep(0.25)
#     else:
#         accel_xout = read_word_2c(0x3b)
#         accel_yout = read_word_2c(0x3d)
#         accel_zout = read_word_2c(0x3f)

#         accel_xout_scaled = accel_xout / 16384.0
#         accel_yout_scaled = accel_yout / 16384.0
#         accel_zout_scaled = accel_zout / 16384.0
#         angle = get_x_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled) 
#         print(angle)

#         EN1.ChangeDutyCycle(100)
#         if angle < -68:     
#             GPIO.output(Motor1['input1'], GPIO.HIGH)
#             GPIO.output(Motor1['input2'], GPIO.LOW)
#         else:     
#             GPIO.output(Motor1['input1'], GPIO.LOW)
#             GPIO.output(Motor1['input2'], GPIO.HIGH)
#         sleep(0.15)
        


