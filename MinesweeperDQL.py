import minesweeper
from minesweeper import MinesweeperEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import copy


# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, 120)   # first fully connected layer
        self.fc2 = nn.Linear(120,100)
        self.fc3 = nn.Linear(100,90)
        self.out = nn.Linear(90, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)       # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class MinesweeperDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.0001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    # ACTIONS = list(range(1, 82))     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the FrozeLake environment
    def train(self, episodes, render=False):
        
        # Create FrozenLake instance
        num_states = 16
        length = 4
        num_actions = 16
        
        epsilon = 0.9 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for i in range(episodes):
            env = MinesweeperEnv()

            state = [random.randint(0,3),random.randint(0,3)]# Initialize to state 0 (top-left corner)
            terminated = False  # True when agent falls in hole or reaches goal
            truncated = False   # True when agent takes more than 200 actions
            #print(state)
            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            #while not terminated and not truncated:
            j = 0
            state, _, terminated, truncated = env.step(state)
            state = state[0]*length + state[1]
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            j=0
            while not terminated and j < 200:
                j+=1
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = [random.randint(0,length-1), random.randint(0,length-1)]
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                        action = [action // length, action % length]

                # Execute action
                if(env.state[action[0],action[1]] in range(0,9)):       
                    continue
                new_state,reward,terminated,truncated = env.step(action)

                #print(env.state)
               #

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state[0]*length + new_state[1]
                #print(reward)
                # Increment step counter
                step_count+=1
            
                #print(reward)
            # Keep track of the rewards collected per episode.
            if 'reward' in locals():
                rewards_per_episode[i] = reward

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "minesweeper_dql.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('minesweeper_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # for _,param in policy_dqn.named_parameters():
            # print(param.data)
        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features
        #print(num_states)

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            
            new_state = new_state[0]*4 + new_state[1]
            
            action = action[0]*4 + action[1]
            #
            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # for _,param in policy_dqn.named_parameters():
        #     print(param.data)

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes):
        # Create Minesweeper environment
        
        num_states = 16
        length = 4
        num_actions = 16

        # Load trained policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("minesweeper_dql.pt"))
        policy_dqn.eval()    # Switch model to evaluation mode

        print('Policy (trained):')
        # self.print_dqn(policy_dqn)  # Uncomment this if needed to print the policy

        wins = 0  # Track number of wins
        total = episodes
        for i in range(episodes):
            env = MinesweeperEnv()
            state = [random.randint(0,3),random.randint(0,3)]# Initialize to state 0 (top-left corner)
            terminated = False  # True when agent falls in hole or reaches goal
            truncated = False   # True when agent takes more than 200 actions
            #print(state)
            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            #while not terminated and not truncated:
            j = 0
            state, _, terminated, truncated = env.step(state)
            state = state[0]*length + state[1]
            while not terminated and j < 200:
                j+=1
                
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                action = [action // length, action % length]  # Convert action to 2D coordinates
                #print(action)
                # Execute action
                if(env.state[action[0],action[1]] in range(0,9)):       
                    continue
                new_state, reward, terminated, truncated = env.step(action)
                #print(env.map)
                #print(reward)
                # print(truncated)

                #print(reward)
                #print(new_state)

                state = new_state[0]*length + new_state[1]
                j+=1

            # Check if the agent won the game (based on how your environment defines a win)
            if 'reward' in locals():
                if reward == 1:  # Assuming a reward of 1 indicates a win
                    wins += 1
            else:
                total -=1

        # Calculate the win rate
        win_rate = wins / total
        print(f"Win rate: {win_rate * 100:.2f}%")


        env.close()

    # Print DQN: state, best action, q values
    # def print_dqn(self, dqn):
    #     # Get number of input nodes
    #     num_states = dqn.fc1.in_features

    #     # Loop each state and print policy to console
    #     for s in range(num_states):
    #         #  Format q values for printing
    #         q_values = ''
    #         for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
    #             q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
    #         q_values=q_values.rstrip()              # Remove space at the end

    #         # Map the best action to L D R U
    #         # best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

    #         # Print policy in the format of: state, action, q values
    #         # The printed layout matches the FrozenLake map.
    #         # print(f'{s:02},{best_action},[{q_values}]', end=' ')         
    #         if (s+1)%4==0:
    #             print() # Print a newline every 4 states

if __name__ == '__main__':
    Minesweeper_Game = MinesweeperDQL()
    Minesweeper_Game.train(50000)
    Minesweeper_Game.test(5000)