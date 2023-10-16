import argparse
import os

import data.split_data as split_data


def arg_init():
    parser = argparse.ArgumentParser(description = 'Training for Edge Reconstruction')
    parser.add_argument('--autosave_every',
                        type = int,
                        default = 100,
                        help = 'Frequency of autosave. Default = 100')
    parser.add_argument('--batch_size',
                        type = int,
                        default = 128,
                        help = 'Batch size for training. Default = 128')
    parser.add_argument('--epochs',
                        type = int,
                        default = 500,
                        help = 'Number of training epochs. Default = 500')
    parser.add_argument('--lambda_fid',
                        type = float,
                        default = 0.5,
                        help = 'Lambda for the fidelity term. Default = 0.5')
    parser.add_argument('--lambda_tv',
                        type = float,
                        default = 0.05,
                        help = 'Lambda for the TV regularizer. Default = 0.05')
    parser.add_argument('--lr',
                        type = float,
                        default = 5e-4,
                        help = 'Learning rate. Default = 5e-4')
    parser.add_argument('--mode',
                        type = str,
                        required = True,
                        choices = ['train', 'test', 'run'],
                        help = 'Train, test or run?')
    parser.add_argument('--model_load_path',
                        type = str,
                        default = '',
                        help = 'Relative path for model to load')
    parser.add_argument('--model_save_dir',
                        type = str,
                        default = 'run1/',
                        help = 'Relative path to directory for model to save')
    parser.add_argument('--n_blocks',
                        type = int,
                        default = 4,
                        help = 'Number of residual blocks. Default = 4')
    parser.add_argument('--n_channels',
                        type = int,
                        default = 32,
                        help = 'Number of channels per residual block. Default = 32')
    parser.add_argument('--shuffle_data',
                        type = int,
                        default = 0,
                        help = 'Shuffle data? Default = 0')
    parser.add_argument('--split_test',
                        type = float,
                        default = 0.15,
                        help = 'Percent of data for testing. Default = 0.15')
    parser.add_argument('--split_train',
                        type = float,
                        default = 0.7,
                        help = 'Percent of data for training. Default = 0.7')
    parser.add_argument('--test_files',
                        type = str,
                        default = 'data/test_files.txt',
                        help = 'Testing image file list')
    parser.add_argument('--threads',
                        type = int,
                        default = 4,
                        help = 'Number of data loader threads')
    parser.add_argument('--thresh',
                        type = float,
                        default = 0.05,
                        help = 'Positive threshold for event triggering [-1,1]. Default = 0.05')
    parser.add_argument('--train_files',
                        type = str,
                        default = 'data/train_files.txt',
                        help = 'Training image file list')
    parser.add_argument('--val_every',
                        type = int,
                        default = 1,
                        help = 'Number of epochs before validation. Default = 1')
    parser.add_argument('--val_files',
                        type = str,
                        default = 'data/val_files.txt',
                        help = 'Validation image file list')
    parser.add_argument('--val_start',
                        type = int,
                        default = 0,
                        help = 'Epoch to start validation. Default = 0')
    args = parser.parse_args()

    # Path resolve
    if args.train_files:
        args.train_files = os.path.join(os.getcwd(), args.train_files)
        args.train_files = docker_resolve(args.train_files)
    if args.test_files:
        args.test_files = os.path.join(os.getcwd(), args.test_files)
        args.test_files = docker_resolve(args.test_files)
    if args.val_files:
        args.val_files = os.path.join(os.getcwd(), args.val_files)
        args.val_files = docker_resolve(args.val_files)
    if args.model_load_path:
        args.model_load_path = os.path.join(os.getcwd(), args.model_load_path)
        args.model_load_path = docker_resolve(args.model_load_path)
    if args.model_save_dir:
        args.model_save_dir = os.path.join(os.getcwd(), 'trained_models', args.model_save_dir)
        args.model_save_dir = docker_resolve(args.model_save_dir)

    # Mode resolve
    args.mode = args.mode.lower()

    # Epoch resolve
    if args.mode == 'test':
        args.epochs = 1

    return args

def docker_resolve(path):
    if 'Edge_Reconstruction' not in path:
        path = os.path.join(path[0], 'Edge_Reconstruction', path[1:])
    return path

###