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
    gyro_xout = read_word_2c(0x43)
    gyro_yout = read_word_2c(0x45)
    gyro_zout = read_word_2c(0x47)

    x_add = gyro_xout/131
    print("{}\t{}\t{}\t{}".format ("X out: ", gyro_xout, "scaled: ", (x_add))) 
    print("{}\t{}\t{}\t{}".format ("Y out: ", gyro_yout, " scaled: ", (gyro_yout / 131))) 
    print("{}\t{}\t{}\t{}".format ("Z out: ", gyro_zout, " scaled: ", (gyro_zout / 131))) 

   
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(x_add)

    xs = xs[-100:]
    ys = ys[-100:]
    ax.clear()
    ax.plot(xs,ys)

# while True:
#     print("Gyroscope data") 
#     print("--------------") 

#     gyro_xout = read_word_2c(0x43)
#     gyro_yout = read_word_2c(0x45)
#     gyro_zout = read_word_2c(0x47)

#     print("{}\t{}\t{}\t{}".format ("X out: ", gyro_xout, "scaled: ", (gyro_xout/131))) 
#     print("{}\t{}\t{}\t{}".format ("Y out: ", gyro_yout, " scaled: ", (gyro_yout / 131))) 
#     print("{}\t{}\t{}\t{}".format ("Z out: ", gyro_zout, " scaled: ", (gyro_zout / 131))) 

#     print()
#     print("Accelerometer data") 
#     print( "------------------")

#     accel_xout = read_word_2c(0x3b)
#     accel_yout = read_word_2c(0x3d)
#     accel_zout = read_word_2c(0x3f)

#     accel_xout_scaled = accel_xout / 16384.0
#     accel_yout_scaled = accel_yout / 16384.0
#     accel_zout_scaled = accel_zout / 16384.0

#     print("{}\t{}\t{}\t{}".format ("X out: ", accel_xout, " scaled: ", accel_xout_scaled)) 
#     print("{}\t{}\t{}\t{}".format ("Y out: ", accel_yout, " scaled: ", accel_yout_scaled)) 
#     print("{}\t{}\t{}\t{}".format ("Z out: ", accel_zout, " scaled: ", accel_zout_scaled)) 
	
#     print()
#     angle = get_x_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)
#     print( "X rotation: ", angle)
#     print("Y rotation: ", get_y_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)) 
    
ani = animation.FuncAnimation(fig,animate,fargs=(xs,ys),interval=100)
plt.show()

    # print()

    # time.sleep(1)