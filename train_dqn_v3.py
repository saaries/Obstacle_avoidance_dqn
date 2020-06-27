import numpy as np
import multiprocessing as mp
import threading
import time
import sys
from functools import partial

from utils.replay_buffer import PrioritizedReplayBuffer
from utils.schedules import LinearSchedule
from env1 import *

INPUT_DIMS = (160, 160, 3)
NUM_ACTIONS = 9
REPLAY_SIZE = 200000
BATCH_SIZE = 32
LEARNING_RATE = 1e-1
GAMMA = 0.99
N_STEPS = 4
GAMMA_N = GAMMA ** N_STEPS

NUM_WORKERS = 16
MAX_STEPS = 10000000
EPSILON_INITIAL = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY_STEPS = 6000000
BETA_INITIAL = 0.4
BETA_FINAL = 1

SAVE_PATH = "./0.001_1_weights.bin"

class Worker(mp.Process):
    def __init__(self, replay_queue, prediction_pipe, epsilon, global_steps):
        mp.Process.__init__(self, daemon = True)
        self.replay_queue = replay_queue
        self.prediction_pipe = prediction_pipe
        self.epsilon = epsilon
        self.global_steps = global_steps

        self.R = 0
        self.buffer = []
        
    def predict(self, s):
        self.prediction_pipe.send(s)
        q = self.prediction_pipe.recv()
        return q
    
    def run_episode(self):
        env = Environment1()
        s = env.reset()
        local_steps = 0
        total_reward = 0
        
        while(True):
            # Choose action
            eps = self.epsilon.value(self.global_steps.value)
            if(np.random.RandomState().uniform() < eps):
                a = np.random.RandomState().randint(9)
            else:
                a = np.argmax(self.predict(s)[0])
            
            # Execute action
            next_s, r, done = env.step(a)
            self.add_replay(s, a, r, next_s, done)
            
            s = next_s
            total_reward += r
            self.global_steps.value += 1
            local_steps += 1
            
            if(done):
                print(self.global_steps.value, local_steps, total_reward)
                s = env.reset()
                total_reward = 0
                local_steps = 0
                break
        
    def add_replay(self, s, a, r, next_s, done):
        if(len(self.buffer) < N_STEPS):
            self.R += r * (GAMMA ** len(self.buffer))
            self.buffer.append((s, a, r, next_s, done))
        else:
            self.buffer.append((s, a, r, next_s, done))
            s_0, a_0, _, _, _ = self.buffer[0]
            _, _, _, s_n, d = self.buffer[N_STEPS - 1]
            self.replay_queue.put((s_0, a_0, self.R, s_n, d))
            self.R = (self.R - self.buffer[0][2] + r * GAMMA_N) / GAMMA
            self.buffer.pop(0)

        if(done):
            while(len(self.buffer) > 0):
                n = len(self.buffer)
                s_0, a_0, _, _, _ = self.buffer[0]
                _, _, _, s_n, d = self.buffer[n - 1]
                self.replay_queue.put((s_0, a_0, self.R, s_n, d))
                self.R = (self.R - self.buffer[0][2]) / GAMMA
                self.buffer.pop(0)
            self.R = 0
    
    def run(self):
        while(True):
            self.run_episode()

if(__name__ == "__main__"):
    import keras
    import keras.backend as K
    import tensorflow as tf
    
    def build_model():
        input = keras.layers.Input(shape = INPUT_DIMS)
        
        x = keras.layers.Conv2D(
            filters = 32,
            kernel_size = 8,
            strides = 4,
            padding = "valid",
            activation = "relu",
            kernel_initializer = keras.initializers.he_normal()
        )(input)
        x = keras.layers.Conv2D(
            filters = 64,
            kernel_size = 4,
            strides = 2,
            padding = "valid",
            activation = "relu",
            kernel_initializer = keras.initializers.he_normal()
        )(x)
        x = keras.layers.Conv2D(
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = "valid",
            activation = "relu",
            kernel_initializer = keras.initializers.he_normal()
        )(x)
        x = keras.layers.Flatten()(x)
        
        q = keras.layers.Dense(
            units = 512,
            activation = "relu",
            kernel_initializer = keras.initializers.he_normal()
        )(x)
        q = keras.layers.Dense(units = 1)(q)
        
        a = keras.layers.Dense(
            units = 512,
            activation = "relu",
            kernel_initializer = keras.initializers.he_normal()
        )(x)
        a = keras.layers.Dense(units = NUM_ACTIONS)(a)
        
        mean = keras.layers.Lambda(lambda x: K.mean(x, axis = -1, keepdims = True))(a)
        a = keras.layers.Subtract()([a, mean])
        q = keras.layers.Add()([q, a])
        
        model = keras.models.Model(inputs = input, outputs = q)
        model._make_predict_function()
        return model
        
    def build_model_for_train(model):
        input = keras.layers.Input(shape = INPUT_DIMS)
        weights = keras.layers.Input(shape = (1,))
        q = model(input)
        
        train_model = keras.models.Model(inputs = [input, weights], outputs = q)
        
        weighted_loss = partial(tf.losses.huber_loss, weights = weights)
        
        train_model.compile(
            optimizer = keras.optimizers.Adam(LEARNING_RATE, decay = 1e-6),
            loss = weighted_loss
        )
        train_model._make_train_function()
        return train_model
        
    def predictQ(s):
        s = s / 128.0 - 1
        return model.predict(s)
        
    def predictTargetQ(s):
        s = s / 128.0 - 1
        return target_model.predict(s)
        
    def update_target(tau = 0.01):
        w = model.get_weights()
        wt = target_model.get_weights()
        for i in range(len(w)):
            wt[i] = (1 - tau) * wt[i] + tau * w[i]
        target_model.set_weights(wt)
        
    def train():
        s, a, r, next_s, d, w, idx = replay.sample(BATCH_SIZE, beta.value(global_steps.value))
        qVals = predictQ(s)
        nextQVals = predictQ(next_s)
        targetQVals = predictTargetQ(next_s)
        
        targets = []
        for i in range(BATCH_SIZE):
            t = r[i]
            if(not d[i]):
                next_a = np.argmax(nextQVals[i])
                t += GAMMA_N * targetQVals[i][next_a]
            targets.append(t)
            qVals[i][a[i]] = t
        
        s = s / 128.0 - 1
        loss = train_model.train_on_batch([s, w], qVals)
        
        qVals = predictQ(s)
        priorities = [abs(targets[i] - qVals[i][a[i]]) + 1e-6 for i in range(BATCH_SIZE)]
        replay.update_priorities(idx, priorities)
        
    class PredictionServer(threading.Thread):
        def __init__(self, pipes):
            threading.Thread.__init__(self)
            self.pipes = pipes
            self.should_stop = False
        
        def run(self):
            while(not self.should_stop):
                time.sleep(0.001)
                for p in pipes:
                    if(p[0].poll()):
                        s = p[0].recv()
                        p[0].send(predictQ(s))
                        
        def stop(self):
            self.should_stop = True
            
    class PullServer(threading.Thread):
        def __init__(self, replay_queue):
            threading.Thread.__init__(self)
            self.replay_queue = replay_queue
            self.should_stop = False
        
        def run(self):
            while(True):
                time.sleep(0.001)
                while((not self.should_stop) and (not replay_queue.empty())):
                    s, a, r, next_s, done = replay_queue.get()
                    with replay_lock:
                        replay.add(s, a, r, next_s, done)
                        
        def stop(self):
            self.should_stop = True
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    K.set_session(session)
    K.manual_variable_initialization(True)

    model = build_model()
    train_model = build_model_for_train(model)
    target_model = build_model()
    session.run(tf.global_variables_initializer())
    target_model.set_weights(model.get_weights())
    epsilon = LinearSchedule(EPSILON_DECAY_STEPS, EPSILON_FINAL, EPSILON_INITIAL)
    beta = LinearSchedule(MAX_STEPS, BETA_FINAL, BETA_INITIAL)
    replay = PrioritizedReplayBuffer(REPLAY_SIZE, alpha = 0.6)
    replay_lock = threading.Lock()

    replay_queue = mp.Queue(maxsize = 100000)
    global_steps = mp.Value("i", 0)
    pipes = [mp.Pipe() for i in range(NUM_WORKERS)]
    prediction_server = PredictionServer(pipes)
    pull_server = PullServer(replay_queue)
    workers = [Worker(replay_queue, pipes[i][1], epsilon, global_steps) for i in range(NUM_WORKERS)]
    prediction_server.start()
    pull_server.start()
    [w.start() for w in workers]
    
    train_steps = 0
    while(global_steps.value < MAX_STEPS):
        try:
            if((global_steps.value > 50000) and (global_steps.value % 4 == 0)):
                with replay_lock:
                    train()
                train_steps += 1
                if(train_steps % 10 == 0):
                    update_target()
                if(train_steps % 1000 == 0):
                    model.save_weights(SAVE_PATH)
        except KeyboardInterrupt:
            model.save_weights(SAVE_PATH)
            sys.exit(1)
    
    model.save_weights(SAVE_PATH)
    [w.terminate() for w in workers]
    prediction_server.stop()
    pull_server.stop()
    prediction_server.join()
    pull_server.join()
