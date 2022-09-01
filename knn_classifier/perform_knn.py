import numpy as np
import torch
import agnostic_omr_dataloader as dl
from data_management.vocabulary import Vocabulary
from torch.utils.data import IterableDataset, DataLoader
import torch
from sklearn import preprocessing, decomposition, neighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import torch.nn as nn 
import training_helper_functions as tr_funcs
import time


model = torch.load("knn_classifier\embedding_model_(2022.08.31.22.43).pt")