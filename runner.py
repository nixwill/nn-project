#!/usr/bin/env python3

import argparse
import json

from nn_project.train import train


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str)
parser.add_argument('--data_limit', type=int)
parser.add_argument('--data_offset', type=int)
parser.add_argument('--validation_split', type=float)
parser.add_argument('--vocab_size', type=int)
parser.add_argument('--padding', type=str)
parser.add_argument('--cell_type', type=str, choices=['simple', 'gru', 'lstm'])
parser.add_argument('--input_embedding_size', type=int)
parser.add_argument('--context_vector_size', type=int)
parser.add_argument('--output_embedding_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('--early_stopping', type=int)
parser.add_argument('--save_models', type=bool)
parser.add_argument('--save_logs', type=bool)
parser.add_argument('--queue_size', type=int)
args = parser.parse_args()

with open('config/default.json') as i:
    hyperparams = json.load(i)

if args.config is not None:
    with open(args.config) as i:
        hyperparams.update(json.load(i))

hyperparams.update({key: value for key, value in vars(args).items() if value is not None})

if 'config' in hyperparams:
    del hyperparams['config']

print('Hyperparameter settings:')
print(json.dumps(hyperparams, indent=4, sort_keys=True))

train(**hyperparams)
