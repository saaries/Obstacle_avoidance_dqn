import math
from environment import *

class Environment1(Environment):
    def __init__(self):
        Environment.__init__(self)
    
    def _generate_obstacles(self):
        if((not self.done) and (self.time % 12 == 0)):

            offset = math.sin(self.time / 1440.0 * math.pi * 2) * math.pi * 2


            for i in range(8):
                angle = math.pi * 2 * i / 8 + offset
                # angle = pi/4 *i + offset
                o1 = Obstacle()
                o1.x = 200
                o1.y = 200
                o1.vx = math.cos(angle) * 90
                o1.vy = math.sin(angle) * 90
                self.obstacles.append(o1)
                
                angle = math.pi * 2 * i / 8 - offset
                o2 = Obstacle()
                o2.x = 200
                o2.y = 200
                o2.vx = math.cos(angle) * 60
                o2.vy = math.sin(angle) * 60
                self.obstacles.append(o2)
                
    def reset(self):
        self.time = 0
        self.obstacles = []
        self.done = False
        self.agent.reset()
        
        # Skip first 60 frames
        for i in range(60):
            self._update(0)
        
        observation, _, _ = self.step(0)
        return observation