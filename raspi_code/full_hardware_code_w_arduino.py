import time
import serial
import adafruit_us100
ser = serial.Serial("/dev/ttyACM0", baudrate=9600, timeout=.01)
ser.reset_input_buffer()

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Motors:
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
Motor1 = {'EN': 25, 'input1': 24, 'input2': 23}
for x in Motor1:
    GPIO.setup(Motor1[x], GPIO.OUT)

EN1 = GPIO.PWM(Motor1['EN'], 100)
EN1.start(0)

# Functions for running sims

def step(observation, thresh_x, thresh_rad):
    terminated = bool(
        observation[0] < -thresh_x
        or observation[0] > thresh_x
        # or -thresh_rad-0.1 < observation[2] < -thresh_rad
        # or thresh_rad+0.1 >observation[2] > thresh_rad
    )
    if not terminated:
        add = 1
    else:
        add = 0

    return add, terminated

def run_motor(action):
    if action == 0:
        print("Going left")
        EN1.ChangeDutyCycle(100)
        GPIO.output(Motor1['input1'], GPIO.HIGH)
        GPIO.output(Motor1['input2'], GPIO.LOW)
        sleep(0.01)
    if action == 1:
        print("going right")
        EN1.ChangeDutyCycle(100)
        GPIO.output(Motor1['input1'], GPIO.LOW)
        GPIO.output(Motor1['input2'], GPIO.HIGH)
        sleep(0.01)        

# Variables

n_actions = 2
n_observations = 4
filefolder = "/home/gracek1459/Documents/L46/L46_sim2Real/"
model = DQN(n_observations, n_actions)
model.load_state_dict(torch.load(filefolder+"cart_pole_policy_lighter.pt", map_location='cpu'))
model.eval()

starting_time = time.time()
count = 0
check = False
threshold_x = 0.3
threshold_rad = 0.25


while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        try:
            values = [float(x) for x in line.split(" , ")]
            # ind0 = x_1_1 reading distance sensor 1 in [mm]
            # ind1 = x_1_2 reading distance sensor 1 in [mm]
            # ind2 = cart position - calc from distance sensor 1 [m]
            # ind3 = cart velocity - calc from distance sensor 1 [m]
            # ind4 = x_2_1 reading distance sensor 2 in mm (50 sec delay) [mm]
            # ind5 = x_2_2 reading distance sensor 2 in mm (50 sec delay) [mm]
            # ind6 = cart position - calc from distance sensor 2 (50 sec delay) [m]
            # ind7 = cart velocity - calc from distance sensor 2 (50 sec delay) [m]
            # ind8 = angle in x - from IMU in [rad]
            # ind9 = angle velocity in x - from gyro in [rad/s]

            # TRY DIFF OBS: 
            ##### observations are always cart position [m], cart velocity [m/s], pole angle [rad], pole angular velocity [rad/s]
            
            #observation = [values[2], values[3], values[8], values[9]]
            #observation = [values[6], values[7], values[8], values[9]]
            print(values)
            observation = [(values[2]+values[6])/2, (values[3]+values[7])/2, values[8]*2, values[9]]
            if check:
                reward, terminated= step(observation,threshold_x,threshold_rad)
                count = count + reward

                print(observation)
                if terminated:
                    print("PENDULUM DIED STOP SCRIPT")
                    break

            #elapsed_time = time.time()
            #print(elapsed_time - starting_time, values)
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = model(state).max(1).indices.view(1, 1)
                # if count%2 == 1: # or observation[2] < 0:
                #     action = 0
                # elif count%2 == 0: # or observation[2] > 0:
                #     action = 1
                
                run_motor(action)
                check = True
        except UnicodeDecodeError:
            print("")
        except IndexError:
            print("")
        except ValueError:
            print("")
        