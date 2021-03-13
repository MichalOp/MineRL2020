# Simple env test.
import json
import select
import time
import logging
import os
import sys

import gym
import minerl
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

# In ONLINE=True mode the code saves only the final version with early stopping,
# in ONLINE=False it saves 20 intermediate versions during training.
ONLINE = True

trains_loaded = True
try:
    from clearml import Task
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
print(MINERL_DATA_ROOT, file=sys.stderr)


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
    torch.set_num_threads(1)
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
            if ONLINE:
                torch.save(model.state_dict(),"train/model.tm")
            else:
                torch.save(model.state_dict(),f"testing/model_{count//int(steps/20)}.tm")
            modcount -= int(steps/20)
            if ONLINE:
                if count//int(steps/20) == 14:
                    break

        if step % 40 == 0:
            print(losssum, count, count/(time()-t0), file=sys.stderr)
            if step > 50 and trains_loaded and not ONLINE:
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
    # a bit of code that creates clearml logging (formerly trains) if clearml
    # is available
    if trains_loaded and not ONLINE:
        task = Task.init(project_name='MineRL', task_name='kmeans pic+pic+tre 1024 + flips whatever')
        logger = task.get_logger()
    else:
        logger = None

    cached_kmeans("train","MineRLObtainDiamondVectorObf-v0")
    print("lets gooo", file=sys.stderr)

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
    
    # Training 100% Completed
    torch.save(model.state_dict(),"train/model.tm")
    print("ok", file=sys.stderr)

    loader.kill()
    #env.close()


if __name__ == "__main__":
    main()
