import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score
import argparse
import csv
from transformers import AutoConfig, AutoModelForSequenceClassification

path_to_biobert = 'nghuyong/ernie-2.0-large-en'
usemoco = True
if usemoco:
    config = AutoConfig.from_pretrained(
            path_to_biobert,
            num_labels=1024,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        path_to_biobert,
        config=config,
    )

    checkpoint = torch.load('./moco_model/moco.tar')
    print(checkpoint.keys())
    print(checkpoint['arch'])
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        if 'module.encoder_q' in key:
            new_key = key[17:]
            state_dict[new_key] = state_dict[key]
        del state_dict[key]
    for key in list(state_dict.keys()):
        if key == 'classifier.0.weight':
            new_key = 'classifier.weight'
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if key == 'classifier.0.bias':
            new_key = 'classifier.bias'
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if key == 'classifier.2.weight' or key == 'classifier.2.bias':
            del state_dict[key]
    state_dict['classifier.weight'] = state_dict['classifier.weight'][:1024, :]
    state_dict['classifier.bias'] = state_dict['classifier.bias'][:1024]
    model.load_state_dict(checkpoint['state_dict'])
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, 2)
    torch.save(model.state_dict(), "./moco_model/moco.p")
    print('finished')
