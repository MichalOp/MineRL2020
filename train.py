# Simple env test.
import json
import select
import time
import logging
import os

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
from time import time
from loader import BatchSeqLoader, absolute_file_paths
from math import sqrt
from kmeans import cached_kmeans


trains_loaded = True
try:
    from trains import Task
except:
    trains_loaded = False
from random import shuffle

from minerl.data import DataPipeline

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
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

BATCH_SIZE = 4
SEQ_LEN = 100

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
        spatial, nonspatial, prev_action, act, _, _, hidden = loader.get_batch(BATCH_SIZE)
        count += BATCH_SIZE*SEQ_LEN
        modcount += BATCH_SIZE*SEQ_LEN
        if mode != "pretrain":
            loss, ldict, hidden = model.get_loss(spatial, nonspatial, prev_action, hidden, torch.zeros(act.shape, dtype=torch.float32, device="cuda"), act)
        else:
            loss, ldict, hidden = model.get_loss(spatial, nonspatial, prev_action, hidden, act, act)

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
            torch.save(model.state_dict(),"train/model.tm")
            torch.save(model.state_dict(),f"testing/model_{count//int(steps/20)}.tm")
            modcount -= int(steps/20)
            # if count//int(steps/20) == 15:
            #     break

        if step % 40 == 0:
            print(losssum, count, count/(time()-t0))
            aicrowd_helper.register_progress(count/steps)
            if step > 50 and trains_loaded:
                for k in loss_dict:
                    logger.report_scalar(title='Training_'+mode, series='loss_'+k, value=loss_dict[k]/40, iteration=int(count)) 
                logger.report_scalar(title='Training_'+mode, series='loss', value=losssum/40, iteration=int(count))
                logger.report_scalar(title='Training_'+mode, series='grad_norm', value=gradsum/40, iteration=int(count))
                logger.report_scalar(title='Training_'+mode, series='learning_rate', value=float(optimizer.param_groups[0]["lr"]), iteration=int(count))
            losssum = 0
            gradsum = 0
            loss_dict = None
            if mode == "fit_selector":
                torch.save(model.state_dict(),"train/model_fitted.tm")
            else:
                torch.save(model.state_dict(),"train/model.tm")

def main():
    if trains_loaded:
        task = Task.init(project_name='MineRL', task_name='kmeans pic+pic+tre 1024 + flips whatever')
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
    train_files = absolute_file_paths('data/MineRLObtainIronPickaxeVectorObf-v0')+\
                  absolute_file_paths('data/MineRLObtainIronPickaxeVectorObf-v0')+\
                  absolute_file_paths('data/MineRLTreechopVectorObf-v0')

    model = Model()
    shuffle(train_files)
    
    loader = BatchSeqLoader(16, train_files, SEQ_LEN, model)
    
    if LOAD:
        model.load_state_dict(torch.load("train/model.tm"))
    model.cuda()
    
    train(model, "train", 150000000, loader, logger)

    # model.selector = Selector()
    # model.selector.cuda()
    # aicrowd_helper.register_progress(0.5)

    # train(model, "fit_selector", 50000000, loader)

        # Print the POV @ the first step of the sequence
        #print(current_state['pov'][0])

        # Print the final reward pf the sequence!
        #print(reward[-1])

        # Check if final (next_state) is terminal.
       # print(done[-1])

        # ... do something with the data.
        #print("At the end of trajectories the length"
              #"can be < max_sequence_len", len(reward))

    # Sample code for illustration, add your training code below
    #env = gym.make(MINERL_GYM_ENV)

#     actions = [env.action_space.sample() for _ in range(10)] # Just doing 10 samples in this example
#     xposes = []
#     for _ in range(1):
#         obs = env.reset()
#         done = False
#         netr = 0

#         # Limiting our code to 1024 steps in this example, you can do "while not done" to run till end
#         while not done:

            # To get better view in your training phase, it is suggested
            # to register progress continuously, example when 54% completed
            # aicrowd_helper.register_progress(0.54)

            # To fetch latest information from instance manager, you can run below when you want to know the state
            #>> parser.update_information()
            #>> print(parser.payload)
            # .payload: provide AIcrowd generated json
            # Example: {'state': 'RUNNING', 'score': {'score': 0.0, 'score_secondary': 0.0}, 'instances': {'1': {'totalNumberSteps': 2001, 'totalNumberEpisodes': 0, 'currentEnvironment': 'MineRLObtainDiamond-v0', 'state': 'IN_PROGRESS', 'episodes': [{'numTicks': 2001, 'environment': 'MineRLObtainDiamond-v0', 'rewards': 0.0, 'state': 'IN_PROGRESS'}], 'score': {'score': 0.0, 'score_secondary': 0.0}}}}
            # .current_state: provide indepth state information avaiable as dictionary (key: instance id)

    # Save trained model to train/ directory
    # Training 100% Completed
    torch.save(model.state_dict(),"train/model.tm")
    print("ok")
    aicrowd_helper.register_progress(1)
    aicrowd_helper.training_end()

    loader.kill()
    #env.close()


if __name__ == "__main__":
    main()
