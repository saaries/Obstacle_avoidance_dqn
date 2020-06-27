import math
import numpy as np
import cv2

'''
Actions:
8 1 2
7 0 3
6 5 4
'''
action_map = [
    (0, 0),(0, -1),(1, -1),
    (1, 0),(1, 1),(0, 1),
    (-1, 1),(-1, 0),(-1, 1)]

WIDTH = 400
HEIGHT = 400
R = 4
speed = 1.5
dt = 1 / 60.0

class EnvObject:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.r = R
        self.time = 0
    
    def update(self):
        self.time += 1
        
class Obstacle(EnvObject):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.vx = 0
        self.vy = 0
        
    def update(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.time += 1
        
class Agent(EnvObject):
    def __init__(self, init_x, init_y):
        super(Agent, self).__init__()
        self.x = init_x
        self.y = init_y
        self.init_x = init_x
        self.init_y = init_y
    
    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.time = 0

class Environment:
    def __init__(self):
        self.time = 0
        self.done = False
        self.obstacles = []
        self.agent = Agent(200, 350)
        self.count = 0
        
    def render(self):
        img = np.full((400, 400), 128, dtype = np.uint8)
        for o in self.obstacles:
            cv2.circle(img, (int(o.x), int(o.y)), int(o.r),\
                       0, -1, lineType = cv2.LINE_AA)
        cv2.circle(img, (int(self.agent.x), int(self.agent.y)),\
                   int(self.agent.r), 255, -1, lineType = cv2.LINE_AA)
        
        return img
        
    def step(self, action):
        frames = []
        reward = 0.0
        
        for i in range(3):
            self._update(action)
            img = self.render()
            cv2.imwrite("img/{}.bmp".format(self.count), img)
            self.count += 1
            
            img = cv2.resize(img, (160, 160))
            frames.append(img)
            if(not self.done):
                reward += 0.001

        if(self.done and (self.time < 3600)):
            reward -= 1.0
            
        observation = np.expand_dims(np.stack(frames, axis = -1), axis = 0)
        return observation, reward, self.done
        
    def reset(self):
        self.time = 0
        self.obstacles = []
        self.done = False
        self.agent.reset()
        
        observation, _, _ = self.step(0)
        return observation
        
    def _generate_obstacles(self):
        pass
        
    def _update(self, action):
        if(self.done):
            return
            
        self._generate_obstacles()
        
        # Update obstacles
        objs = []
        for o in self.obstacles:
            o.update()
            if((o.x + o.r >= 0) and (o.x - o.r <= 400)\
               and (o.y + o.r >= 0) and (o.y - o.r <= 400)):
                objs.append(o)
        self.obstacles = objs
        
        # Update agent
        dx, dy = action_map[action]
        if((dx == -1) and (self.agent.x <= self.agent.r)):
            dx = 0
        if((dx == 1) and (self.agent.x >= 400 - self.agent.r)):
            dx = 0
        if((dy == -1) and (self.agent.y <= self.agent.r)):
            dy = 0
        if((dy == 1) and (self.agent.y >= 400 - self.agent.r)):
            dy = 0
        if((dx != 0) and (dy != 0)):
            dx *= 0.707
            dy *= 0.707
        self.agent.x += dx * speed
        self.agent.y += dy * speed
        
        # Detect collision
        for o in self.obstacles:
            d = (self.agent.x - o.x) ** 2 + (self.agent.y - o.y) ** 2
            if(d <= (self.agent.r + o.r) ** 2):
                # End of episode
                self.done = True
                break
        
        self.time += 1
        if(self.time >= 3600):
            self.done = True
