import gym
from gym import spaces
import numpy as np
from random import randrange
import pygame
import pygame.freetype

class MinesweeperEnv(gym.Env):

    # a minesweeper environment implemented in openai gym style
    # STATE
    #   given the map of height x width, numbers in each block of map:
    #   1. -1 denotes unopened block
    #   2. 0-8 denotes the number of mines in the surrounding 8 blocks of the block
    # MAP
    #   the map is unobservable to the agent where 0 is non-mine, 1 has a mine
    # ACTIONS
    #   action is Discrete(height*width) representing an attempt to open at the block's
    #   index when the map is flattened.
    # RESET
    #   be sure to call the reset function before using any other function in the class
    # STEP(ACTION)
    #   returns a four tuple: next_state, reward, done, _
    # RENDER
    #   renders the current state using pygame

    def __init__(self, height=4, width=4, num_mines=3):
        self.observation_space = spaces.Box(-1, 8, shape=(height,width), dtype=int)
        self.action_space = spaces.MultiDiscrete([height,width])

        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.win_reward = 1
        self.fail_reward = -1
        self.pos_prog = 0.9
        self.neg_prog = -0.3
        self.map = np.array([[False]*width for _ in range(height)])
        self.state = np.zeros((height, width),dtype=int)-1
        self.step_cntr = 0
        self.step_cntr_max = (height*width-num_mines)*2

        self.block_size = 25
        self.window_height = self.block_size * height
        self.window_width = self.block_size * width
        self.map = None
        self.generate_mines()
        
        self.screen = None

    def generate_mines(self):
        self.map = np.array([[False]*self.width for _ in range(self.height)])
        for _ in range(self.num_mines):
            x = randrange(self.height)
            y = randrange(self.width)
            while self.map[x,y]:
                x = randrange(self.height)
                y = randrange(self.width)
            self.map[x,y] = True

    def reset(self, seed=None, return_info=False, options=None):
        self.generate_mines()
        self.step_cntr = 0
        self.state = np.zeros((self.height, self.width),dtype=int)-1
        return self.state

    def get_num_opened(self):
        count = 0
        for i in self.state.flatten():
            if i>=0: count += 1
        return count

    def get_num_surr(self, x, y):
        count = 0
        for i in range(max(0,x-1), min(self.height,x+2)):
            for j in range(max(0,y-1), min(self.width,y+2)):
                if not (i==x and j==y):
                    if self.map[i,j]:
                        count += 1
        return count

    def update_state(self, x, y):
        # print(x, y)
        num_surr = self.get_num_surr(x,y)
        self.state[x,y] = num_surr
        # print(self.state[x,y])
        if num_surr==0:
            for i in range(max(0,x-1), min(self.height,x+2)):
                for j in range(max(0,y-1), min(self.width,y+2)):
                    if (not (i==x and j==y)) and self.state[i,j]==-1:
                        self.update_state(i,j)

    def step(self, action):
        if len(action)!=2 or action[0]<0 or action[1]>=self.height or action[1]<0 or action[1]>=self.width:
            raise ValueError
        info = self._get_info()
        # if self.step_cntr==self.step_cntr_max:
        #     return self.state, 0, True, info
        # else:
        #     self.step_cntr += 1
        x,y = action[0],action[1]

        # Reward Structure
        if self.map[x][y]:
            self.state[x,y] = 9
            return [x,y], self.fail_reward, True, info
        else:
            num_opened = self.get_num_opened()
            # if self.state[x,y]!=-1:
            #     return self.state, 0, False, info
            oldstate = (self.state[x,y]).copy()
            surr = [True for i in range(max(0,x-1), min(self.height,x+2)) for j in range(max(0,y-1), min(self.width,y+2)) if self.state[i,j]==-1]
            #print(surr)
            if(False not in surr):
                flag = 1
            else:
                flag = 0
            self.update_state(x,y)
            new_num_opened = self.get_num_opened()
            
            if new_num_opened==self.height*self.width-self.num_mines:
                #print("ENTERS1")
                return [x,y], self.win_reward, True, info
            if oldstate in range(0, 9):
                #print("ENTERS4")
                return [x,y], self.neg_prog, False, info
            if oldstate== -1:
                if(flag == 1):
                    return [x,y], -0.3, False, info
                else:
                    #print("ENTERS3")
                    return [x,y], self.pos_prog, False, info
            
                
            return [x,y], new_num_opened-num_opened, False, info

    def drawGrid(self):
        for y in range(0, self.window_width, self.block_size):
            for x in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                num = int(self.state[int(x/self.block_size),int(y/self.block_size)])
                if num==-1:
                    pygame.draw.rect(self.screen, (255,255,255), rect, 1)
                else:
                    color = (250, 250-num*30, 250-num*30)
                    pygame.draw.rect(self.screen, color, rect)
                    text = self.font.get_rect(str(num))
                    text.center = rect.center
                    self.font.render_to(self.screen,text.topleft,str(num),(0,0,0))
        pygame.display.update()

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.font = pygame.freetype.SysFont(pygame.font.get_default_font(), 13)
        self.screen.fill((0,0,0))
        self.drawGrid()

    def _get_info(self):
        return {"map":self.map}
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()