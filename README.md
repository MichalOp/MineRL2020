# Using supervised learning to train an RNN that sometimes tries to do something of value in Minecraft 
This repository contains my somewhat cleaned up solution for MineRL 2020 that has reached the 3rd place (1st in pure imitation category).
The quality of the code is not particularly great, but as it is quite short, it might still be helpful as a starting point in the future editions.

## The code
Written in PyTorch. 
* The main training loop is located in ```train.py```. 
* ```loader.py``` contains BatchSeqLoader, which keeps a number of open replays
and randomly selects a subset of them to load a batch from at each training step.
* ```model.py``` contains the neural network.
* ```test.py``` evaluates the trained neural network in the environment.

## The architecture
* FixUp ResNet for visual processing, copied from [https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/model.py](https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/model.py)
* LSTM with 1024 units, trained by iterating over training sequences of length 100, without skipping frames
* K-Means with K=120 to discretize the action space, similar to the baselines
* Standard supervised classification with cross entropy - network predicts what a player would do given some string of observations
* Uses ObtainIronPickaxe and Treechop datasets (ObtainDiamond was seemingly too noisy and reduced performance)
* Data augmentation - transposing and flipping the images, permuting color channels
 
## The peformance
The trained network should achieve average scores of around 11-16 after 200 episodes. 

## Running
* To train the network, run ```train.py``` - it should take around 15h on a high-end gpu.
* To run the evaluation, run ```test.py```, possibly changing the number of evaluation threads - 4 threads have okay-ish performance on a 16-thread CPU.