from minerl.data import DataPipeline
import threading as mp
from itertools import cycle
from minerl.data.util import minibatch_gen
import minerl
import torch
from random import shuffle, random
import os
import sys
from kmeans import cached_kmeans
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from queue import Queue

class PPipeEnd:
    ''' An multiprocessing.Pipe emulator for threading, developed as a quick fix
    when it turned out multiprocessing doesn't work on evaluation servers.'''
    def __init__(self, in_q, out_q):
        self.in_q = in_q
        self.out_q= out_q

    def send(self,data):
        self.out_q.put(data)
    
    def recv(self):
        return self.in_q.get()


def pseudo_pipe():
    q1 = Queue()
    q2 = Queue()

    return PPipeEnd(q1, q2), PPipeEnd(q2, q1)


def loader(files, pipe, main_sem, internal_sem, batch_size):
    torch.set_num_threads(1)
    kmeans = cached_kmeans("train","MineRLObtainDiamondVectorObf-v0")

    files = cycle(files)

    while True:
        f = next(files)
        
        try:
            d = DataPipeline._load_data_pyfunc(f, -1, None)
        except:
            continue
        pipe.send("RESET")
        steps = 0
        obs, act, reward, nextobs, done = d
        obs_screen = torch.tensor(obs["pov"], dtype=torch.float32).transpose(1,3).transpose(2,3)
        obs_vector = torch.tensor(obs["vector"], dtype=torch.float32)
        flip_data = torch.ones((obs_vector.shape[0], 2), dtype=torch.float32)

        if random() > 0.5:
            obs_screen = torch.flip(obs_screen, [2])
            flip_data[:,0] = -1
        
        if random() > 0.5:
            obs_screen = obs_screen.transpose(2,3)
            flip_data[:,1] = -1

        if random() > 0.5:
            obs_screen = torch.flip(obs_screen, [1])

        obs_vector = torch.cat([obs_vector, flip_data], dim=1)

        running = 1 - torch.tensor(done, dtype=torch.float32)
        rewards = torch.tensor(reward, dtype=torch.float32)
        encoded = kmeans.predict(act["vector"])
        actions = torch.tensor(encoded, dtype=torch.int64)
        prev_action = torch.cat([torch.zeros((1,),dtype=torch.int64), actions[:-1]], dim=0)
        l = actions.shape[0]
        for i in range(0, l, batch_size):
            steps += 1
            
            if l - i < batch_size:
                break
            
            internal_sem.release()
            main_sem.release()

            msg = pipe.recv()
            if msg == "GET":
                pass
            elif msg == "STOP":
                print("Shutting down", file=sys.stderr)
                return

            pipe.send((obs_screen[i:i+batch_size], obs_vector[i:i+batch_size], prev_action[i:i+batch_size], actions[i:i+batch_size], running[i:i+batch_size], rewards[i:i+batch_size]))


class ReplayRoller():

    def __init__(self, files_queue, model, sem, batch_size, prefetch):
        self.batch_size = batch_size
        self.sem = sem
        self.model = model
        self.in_sem = mp.Semaphore(0)
        self.data = []
        self.hidden = self.model.get_zero_state(1)
        #print(self.hidden)
        self.hidden = (self.hidden[0].cuda(),self.hidden[1].cuda())
        self.pipe_my, pipe_other = pseudo_pipe()
        self.files = files_queue
        self.loader = mp.Thread(target=loader,args=(self.files,pipe_other,self.sem,self.in_sem, self.batch_size))
        self.loader.start()


    def get(self):
        
        if not self.in_sem.acquire(blocking=False):
            return []

        self.pipe_my.send("GET")
        data = self.pipe_my.recv()

        while data == "RESET":
            self.hidden = self.model.get_zero_state(1)
            self.hidden = (self.hidden[0].cuda(),self.hidden[1].cuda())
            data = self.pipe_my.recv()

        return data + (self.hidden,)

    def kill(self):
        self.pipe_my.send("STOP")
        self.loader.join()

    def set_hidden(self, new_hidden):
        self.hidden = new_hidden


class BatchSeqLoader():
    '''
    This loader attempts to diversify loaded samples by keeping a pool of open
    replays and randomly selecting several to load a sequence from at each training
    step.
    '''

    def __init__(self, envs, names, steps, model):
        self.main_sem = mp.Semaphore(0)
        self.rollers = []
        def chunkIt(seq, num):
            avg = len(seq) / float(num)
            out = []
            last = 0.0

            while last < len(seq):
                out.append(seq[int(last):int(last + avg)])
                last += avg

            return out

        names = chunkIt(names, envs)

        for i in range(envs):
            self.rollers.append(ReplayRoller(names[i], model, self.main_sem, steps, 1))    
    
    def batch_lstm(self,states):
        states = zip(*states)
        return tuple([torch.cat(s,1) for s in states])

    def unbatch_lstm(self,state):
        l = state[0].shape[1]
        output = []
        for i in range(l):
            output.append((state[0][:,i:i+1].detach(), state[1][:,i:i+1].detach()))

        return output

    def get_batch(self, batch_size):

        shuffle(self.rollers)
        data, self.current_rollers = [],[]
        while len(data) < batch_size:
            self.main_sem.acquire()
            for roller in self.rollers:
                maybe_data = roller.get()
                if len(maybe_data) > 0:
                    sample = maybe_data
                    data.append(sample)
                    self.current_rollers.append(roller)
                    if len(data) == batch_size:
                        break

        data = list(zip(*data))
        output = []
        for d in data[:-1]:
            padded = pad_sequence(d).cuda()
            #print(d[0].shape)
            #print(padded.shape)
            output.append(padded)

        return output + [self.batch_lstm(data[-1])]

    def put_back(self, lstm_state):
        lstm_state = self.unbatch_lstm(lstm_state)
        for i, roller in enumerate(self.current_rollers):
            roller.set_hidden(lstm_state[i])

    def kill(self):
        for roller in self.rollers:
            roller.kill()

class dummy_model:
    def get_zero_state(self, x):
        return (torch.zeros((1,1,1)),torch.zeros((1,1,1)))


def absolute_file_paths(directory):
    return [os.path.join(directory, path) for path in os.listdir(directory)]


if __name__ == "__main__":
    data = minerl.data.make('MineRLObtainDiamondVectorObf-v0', data_dir='data/',num_workers=6)
    model = dummy_model()
    loader = BatchSeqLoader(1, data._get_all_valid_recordings('data/MineRLObtainDiamondVectorObf-v0'), 128, model)
    i = 0
    while True:
        i+=1
        print(i)
        _,_,_,data = loader.get_batch(1)
        loader.put_back(data)