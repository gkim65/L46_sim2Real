import neptune
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame
from IPython import display

####################################################
################    SETUP    ###################
####################################################

#----------------------------------------------------

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#----------------------------------------------------

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#----------------------------------------------------

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

####################################################
################    FUNCTIONS    ###################
####################################################
    
def select_action(state, EPS_END, EPS_START, EPS_DECAY,steps_done,policy_net,env,device,random_gen):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            obs = state.tolist()
            for i in range(len(obs[0])):
                obs[0][i] = obs[0][i] + np.random.uniform(0,random_gen)
            state = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
            return policy_net(state).max(1).indices.view(1, 1), steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), steps_done

def plot_durations(show_result=False, episode_durations = [], is_ipython = True):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model(memory,run,device,policy_net,target_net,optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    print(loss)
    run["train/loss"].append(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # return loss

def run_model(run, num_ep_user, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hardware_prob_response,name,edge_limit,random_gen):
    
    
    env = gym.make("CartPole-v1", render_mode = "human")

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.ones(1, device=device)
        print(x)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print (x)
    else:
        print ("MPS device not found, using CPU")
        device = torch.device("cpu")
        
    if torch.backends.mps.is_available():
        num_episodes = num_ep_user if num_ep_user > 0 else 500
    else:
        num_episodes = num_ep_user if num_ep_user > 0 else 10
    
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    HARDWARE_FORCE_LIMIT = env.force_mag
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAYMEMORY)

    steps_done = 0

    episode_durations = []

    params = {"Batch_size": BATCH_SIZE, 
                "Gamma": GAMMA,
                "EPS_START": EPS_START,
                "EPS_END": EPS_END,
                "EPS_DECAY": EPS_DECAY,
                "TAU": TAU,
                "Learning Rate": LR,
                "n_actions": n_actions,
                "optimizer": "adam",
                "num_episodes": num_episodes,
                "hardware problem response" : hardware_prob_response,
                "hardware problem force_limit" : HARDWARE_FORCE_LIMIT,
                "Test Name" : name,
                "edge_limit": edge_limit,
                "Random_Gen": random_gen
                }
    run["parameters"] = params
    
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        reward_total = 0
        for t in count():
            action,steps_done = select_action(state, EPS_END, EPS_START, EPS_DECAY,steps_done,policy_net,env,device,random_gen)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            if i_episode == num_episodes-1:
                run["last_run_log/state_position"].append(observation[0])
                run["last_run_log/state_velocity"].append(observation[1])
                run["last_run_log/state_angle"].append(observation[2])
                run["last_run_log/state_angle_velocity"].append(observation[3])
                run["last_run_log/last_action"].append(action.item())

            if hardware_prob_response > 0:
                for i in range(hardware_prob_response):
                    observation, reward, terminated, truncated, _ = env.step(action.item())

                    if i_episode == num_episodes-1:
                        run["last_run_log/state_position"].append(observation[0])
                        run["last_run_log/state_velocity"].append(observation[1])
                        run["last_run_log/state_angle"].append(observation[2])
                        run["last_run_log/state_angle_velocity"].append(observation[3])
                        run["last_run_log/last_action"].append(action.item())

                    reward = torch.tensor([reward], device=device)
                    done = terminated or truncated or abs(observation[0]) > edge_limit
                    if terminated:
                        next_state = None
                        break
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                        memory.push(state, action, next_state, reward)
                        # Move to the next state
                        state = next_state
            else:
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated or abs(observation[0]) > edge_limit
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            
            # Store the transition in memory
            reward_total = reward_total + reward
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory,run,device,policy_net,target_net,optimizer)
            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                run["train/episode_durations"].append(t)
                episode_durations.append(t + 1)
                plot_durations(show_result=False, episode_durations=episode_durations)
                break



    run["eval/reward_total"] = reward_total


    print('Complete')
    plot_durations(show_result=False, episode_durations=episode_durations)
    plt.ioff()
    # plt.show()
    pygame.quit()
    env.close()
    run.stop()


if __name__ == "__main__":
    
    ####################################### Variables ###################################### 
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 256
    GAMMA = 0.5
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-3 #1e-4
    REPLAYMEMORY = 10000
    hardware_prob_response = 1
    num_ep_user = 500
    edge_limit = 0.3
    random_gen = 0.05

    test = [0.3]#,0.6,1.2]
    for edge_limit in test:

        name = "Test on - edge_limits"
        run = neptune.init_run(
            project="ACS-MPhil/L46",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YzEzNjgwMi1kMWMzLTRhNGUtYTJkNy0wNWJmNjRhYWIzNWMifQ==",
        )  # your credentials

        run_model(run, num_ep_user, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hardware_prob_response,name,edge_limit,random_gen)


        


    
        
    

