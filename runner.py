#!/usr/bin/env python3

import argparse
import json

from nn_project.train import train


with open('config/default.json') as i:
    default = json.load(i)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=None)
parser.add_argument('--data_limit', type=int, default=default['data_limit'])
parser.add_argument('--data_offset', type=int, default=default['data_offset'])
parser.add_argument('--validation_split', type=float, default=default['validation_split'])
parser.add_argument('--vocab_size', type=int, default=default['vocab_size'])
parser.add_argument('--padding', type=str, default=default['padding'])
parser.add_argument('--cell_type', type=str, default=default['cell_type'], choices=['simple', 'gru', 'lstm'])
parser.add_argument('--input_embedding_size', type=int, default=default['input_embedding_size'])
parser.add_argument('--context_vector_size', type=int, default=default['context_vector_size'])
parser.add_argument('--output_embedding_size', type=int, default=default['output_embedding_size'])
parser.add_argument('--learning_rate', type=float, default=default['learning_rate'])
parser.add_argument('--batch_size', type=int, default=default['batch_size'])
parser.add_argument('-e', '--epochs', type=int, default=default['epochs'])
parser.add_argument('--early_stopping', type=int, default=default['early_stopping'])
parser.add_argument('--save_models', type=bool, default=default['save_models'])
parser.add_argument('--save_logs', type=bool, default=default['save_logs'])
parser.add_argument('--queue_size', type=int, default=default['queue_size'])
args = parser.parse_args()

if args.config is None:
    hyperparams = vars(args)
    del hyperparams['config']
else:
    # TODO fix merging with cmd args
    with open(args.config) as i:
        hyperparams = json.load(i)

print('Hyperparameter settings:')
print(json.dumps(hyperparams, indent=4, sort_keys=True))

train(**hyperparams)
