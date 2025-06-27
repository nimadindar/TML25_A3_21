"""
The credit for this code belongs to the following github repository:
    https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training

This repo is an official implementation of the paper: Geometry-Aware Instance-Reweighted Adversarial Training
by Zhang et. al. (2021). We have used their adversarial attack scheme to directly test and build our other implementations
upon it. 
"""
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.load_model import ModelZoo
import torch
import numpy as np

def GAIR(num_steps, Kappa, Lambda, func):
    # Weight assign
    if func == "Tanh":
        reweight = ((Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).tanh()+1)/2
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Sigmoid":
        reweight = (Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).sigmoid()
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Discrete":
        reweight = ((num_steps+1)-Kappa)/(num_steps+1)
        normalized_reweight = reweight * len(reweight) / reweight.sum()
            
    return normalized_reweight