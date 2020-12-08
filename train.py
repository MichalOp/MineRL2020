# Simple env test.
import json
import select
import time
import logging
import os
import threading

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser
import coloredlogs
coloredlogs.install(logging.DEBUG)

from model import Model
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from time import time, sleep
from loader import BatchSeqLoader, absolute_file_paths
from math import sqrt
from kmeans import cached_kmeans
from collections import deque

trains_loaded = True
try:
    from trains import Task
except:
    trains_loaded = False
from random import shuffle, sample

from minerl.data import DataPipeline

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondDenseVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
print(MINERL_DATA_ROOT)

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

BATCH_SIZE = 8
SEQ_LEN = 30

FIT = True
LOAD = False
FULL = True

def update_loss_dict(old, new):
    if old is not None:
        for k in old:
            old[k] += new[k]
        return old
    return new


def train(model, mode, steps, loader, logger):
    
    if mode != "fit_selector":
        optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-6)
    else:
        optimizer = Adam(params=model.selector.parameters(), lr=1e-4, weight_decay=1e-6)

    def lambda1(x):
        return min((1e-1)* (sqrt(sqrt(sqrt(10)))**min(x, 50)),1)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    optimizer.zero_grad()
    step = 0
    count = 0
    t0 = time()
    losssum = 0
    gradsum = 0
    loss_dict = None
    modcount = 0
    for i in range(int(steps/ BATCH_SIZE / SEQ_LEN)):
        step+=1
        #print(i)
        spatial, nonspatial, prev_action, act, _, rewards, hidden = loader.get_batch(BATCH_SIZE)
        count += BATCH_SIZE*SEQ_LEN
        modcount += BATCH_SIZE*SEQ_LEN
        if mode != "pretrain":
            loss, ldict, hidden = model.get_loss(spatial, nonspatial, prev_action, hidden, torch.zeros(act.shape, dtype=torch.float32, device="cuda"), act, rewards)
        else:
            loss, ldict, hidden = model.get_loss(spatial, nonspatial, prev_action, hidden, act, act, rewards)

        loss_dict = update_loss_dict(loss_dict, ldict)
        loader.put_back(hidden)

        loss = loss.sum() # / BATCH_SIZE / SEQ_LEN
        loss.backward()
        
        losssum += loss.item()
        
        if mode == "fit_selector":
            grad_norm = clip_grad_norm_(model.selector.parameters(),10)
        else:
            grad_norm = clip_grad_norm_(model.parameters(),10)
        
        gradsum += grad_norm.item()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        if modcount >= steps/20:
            #torch.save(model.state_dict(),"train/model.tm")
            torch.save(model.state_dict(),f"testing/model_{count//int(steps/20)}.tm")
            modcount -= int(steps/20)
            #if count//int(steps/20) == 15:
            #    break

        if step % 20 == 0:
            print(losssum, count, count/(time()-t0))
            aicrowd_helper.register_progress(count/steps)
            if step > 50 and trains_loaded:
                for k in loss_dict:
                    logger.report_scalar(title='Training_'+mode, series='loss_'+k, value=loss_dict[k]/20, iteration=int(count)) 
                logger.report_scalar(title='Training_'+mode, series='loss', value=losssum/20, iteration=int(count))
                logger.report_scalar(title='Training_'+mode, series='grad_norm', value=gradsum/20, iteration=int(count))
                logger.report_scalar(title='Training_'+mode, series='learning_rate', value=float(optimizer.param_groups[0]["lr"]), iteration=int(count))
            losssum = 0
            gradsum = 0
            loss_dict = None
            if mode == "fit_selector":
                torch.save(model.state_dict(),"train/model_fitted.tm")
            else:
                torch.save(model.state_dict(),"train/model.tm")


def run_single_env(model, env, obs_queue, stop, reward_tracking):
    counter = 0
    while True:
        reward_sum = 0
        with torch.no_grad():
            obs = env.reset()
            done = False
            state = model.get_zero_state(1, device="cuda")
            old_state = state
            s = torch.zeros((1,1,64), dtype=torch.float32, device="cuda")
            
            buffer = []

            while not done:
                counter += 1
                # if counter > 7900000:
                #     return
                spatial = torch.tensor(obs["pov"], device="cuda", dtype=torch.float32).unsqueeze(0).unsqueeze(0).transpose(2,4)
                nonspatial = torch.tensor(obs["vector"], device="cuda", dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                s, action, state = model.sample(spatial, nonspatial, s, state, torch.zeros((1,1,64),dtype=torch.float32, device="cuda"))
                
                obs,reward,done,_ = env.step({"vector":s})

                buffer.append((spatial, nonspatial, action, torch.tensor([[reward]], device="cuda", dtype=torch.float32)))

                if len(buffer) == SEQ_LEN:
                    sequence = []
                    for d in zip(*buffer):
                        sequence.append(torch.cat(d, dim=0).squeeze(dim=1))
                    
                    sequence.append(old_state)
                    obs_queue.append(sequence)
                    old_state = state
                    buffer = []

                reward_sum += reward

            reward_tracking.append(reward_sum)


def train_rl(model, mode, steps, loader, logger):

    target_model = Model()

    target_model.load_state_dict(model.state_dict())
    target_model.cuda()

    if mode != "fit_selector":
        optimizer = Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-6)
    else:
        optimizer = Adam(params=model.selector.parameters(), lr=1e-5, weight_decay=1e-6)

    def lambda1(x):
        return min((1e-1)* (sqrt(sqrt(sqrt(10)))**min(x, 50)),1)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    optimizer.zero_grad()
    step = 0
    count = 0
    t0 = time()
    losssum = 0
    gradsum = 0
    loss_dict = None
    modcount = 0

    reward_tracking = deque(maxlen=100)

    data = deque(maxlen=3000)
    env = gym.make(MINERL_GYM_ENV)
    t = threading.Thread(target=run_single_env, args=(model, env, data, [False], reward_tracking))
    t.start()
    while len(data) < 500:
        sleep(1)
    for i in range(int(steps/ BATCH_SIZE / SEQ_LEN)*100):
        step+=1
        #print(i)
        spatial, nonspatial, act, rewards, hidden = loader.get_batch(3, additional_data=sample(data, 3))
        count += BATCH_SIZE*SEQ_LEN
        modcount += BATCH_SIZE*SEQ_LEN
        loss, ldict, hidden = model.get_loss_rl(target_model, spatial, nonspatial, hidden, act, rewards)

        loss_dict = update_loss_dict(loss_dict, ldict)

        loss = loss.sum() # / BATCH_SIZE / SEQ_LEN
        loss.backward()
        loader.put_back(hidden)
        losssum += loss.item()
        
        grad_norm = clip_grad_norm_(model.parameters(),10)
        
        gradsum += grad_norm.item()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        target_model.soft_update(model, 0.001)

        if modcount >= steps/20:
            #torch.save(model.state_dict(),"train/model.tm")
            torch.save(model.state_dict(),f"testing/model_{count//int(steps/20)}.tm")
            modcount -= int(steps/20)
            #if count//int(steps/20) == 15:
            #    break

        if step % 20 == 0:
            print(losssum, count, count/(time()-t0), len(data))
            aicrowd_helper.register_progress(count/steps)
            if trains_loaded:
                for k in loss_dict:
                    logger.report_scalar(title='TrainingRL_'+mode, series='loss_'+k, value=loss_dict[k]/20, iteration=int(count)) 
                logger.report_scalar(title='TrainingRL_'+mode, series='loss', value=losssum/20, iteration=int(count))
                logger.report_scalar(title='TrainingRL_'+mode, series='grad_norm', value=gradsum/20, iteration=int(count))
                logger.report_scalar(title='TrainingRL_'+mode, series='reward_avg', value=sum(reward_tracking)/(len(reward_tracking)+0.001), iteration=int(count))
            losssum = 0
            gradsum = 0
            loss_dict = None
            torch.save(model.state_dict(),"train/model_rl.tm")

def main():
    if trains_loaded:
        task = Task.init(project_name='MineRL', task_name='kmeans 120 dia+pic+tre 512 reinforcement learning seq=30 auxilliary tasks dense')
        logger = task.get_logger()
    else:
        logger = None

    aicrowd_helper.training_start()
    cached_kmeans("train","MineRLObtainDiamondVectorObf-v0")
    
    """
    This function will be called for training phase.
    """
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    #data = minerl.data.make('MineRLObtainDiamondVectorObf-v0', data_dir='data/',num_workers=6)
    train_files = absolute_file_paths('data/MineRLObtainDiamondDenseVectorObf-v0')+\
                  absolute_file_paths('data/MineRLObtainIronPickaxeDenseVectorObf-v0')+\
                  absolute_file_paths('data/MineRLTreechopVectorObf-v0')

    model = Model()
    shuffle(train_files)
    
    loader = BatchSeqLoader(16, train_files, SEQ_LEN, model)
    if LOAD:
        model.load_state_dict(torch.load("train/model_rl.tm"))
    model.cuda()
    
    #train(model, "train", 150000000, loader, logger)
    #loader.kill()

    train_rl(model, "train", 150000000, loader, logger)
    # model.selector = Selector()
    # model.selector.cuda()
    # aicrowd_helper.register_progress(0.5)

    #torch.save(model.state_dict(),"train/model.tm")
    print("ok")
    aicrowd_helper.register_progress(1)
    aicrowd_helper.training_end()

    loader.kill()
    #env.close()


if __name__ == "__main__":
    main()
