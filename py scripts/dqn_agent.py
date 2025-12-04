import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from snake_game import SnakeGame, Direction, BLOCK_SIZE

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # learning rate


# Neural Network 
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        torch.save(self.state_dict(), file_name)


# Trainer 
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Converting to tensors
        state      = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action     = torch.tensor(action, dtype=torch.long)
        reward     = torch.tensor(reward, dtype=torch.float)

        # If we have a single sample, add batch dimension
        if len(state.shape) == 1:
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = (done, )

        # 1. Predicted Q values with current state
        pred = self.model(state)  # shape: [batch, 3]

        # 2. Compute target Q values
        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_next = self.model(next_state[idx])
                Q_new = reward[idx] + self.gamma * torch.max(Q_next)

            # action[idx] is the index of the action (0,1,2)
            target[idx][action[idx]] = Q_new

        # 3. Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()


# Agent 
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0        # controls exploration
        self.gamma = 0.9        # discount factor
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)  # 11 inputs -> 3 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        #track how many moves were randoms vs how many exploited
        self.random_moves = 0
        self.model_moves = 0

    # 11-feature state: danger, direction, food location
    def get_state(self, game: SnakeGame):
        head = game.head

        # Points around the head
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        danger_straight = (
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d))
        )

        danger_right = (
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d))
        )

        danger_left = (
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d))
        )

        food_left  = game.food[0] < game.head[0]
        food_right = game.food[0] > game.head[0]
        food_up    = game.food[1] < game.head[1]
        food_down  = game.food[1] > game.head[1]

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),

            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # epsilon-greedy: more random at the start, less later
        self.epsilon = max(0, 80 - self.n_games) 
        
        #exploration
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  
            self.random_moves += 1
        else:
            #exploit
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            self.model_moves +=1

        return move
