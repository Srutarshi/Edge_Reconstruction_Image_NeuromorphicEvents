import os

import torch
import torch.nn as nn

from data.split_data import split_data
import src.ER_Data as ERD
import src.ER_Model as ERM
from src.run_ER_Model import run
import src.misc.checks as checks
from src.misc.arguments import arg_init


# Argument Parsing
print('===> Parsing arguments...')
args = arg_init()
checks.check_args(args = args)
if args.mode == 'train':
	with open(os.path.join(args.model_save_dir, 'args.txt'), 'w') as f:
		f.write(str(args))


# Create Datasets
print('===> Building datasets...')
if args.shuffle_data:
	split_data(split_train = args.split_train,
	           split_test = args.split_test)
datasets = ERD.create_datasets(train_list = args.train_files,
                               test_list = args.test_files,
                               val_list = args.val_files,
                               mode = args.mode,
                               thresh = args.thresh)
checks.check_data(data = datasets,
                  mode = args.mode,
                  check = 'datasets')


# Create Data Loaders
print('===> Creating data loaders...')
data_loaders = ERD.create_dataloaders(datasets = datasets,
                                      n_workers = args.threads,
                                      batch_size = args.batch_size)
checks.check_data(data = data_loaders,
                  mode = args.mode,
                  check = 'dataloaders')


# Instantiate Model
print('===> Instantiating model...')
model = ERM.build_model(mode = args.mode,
                        path = args.model_load_path,
                        lr = args.lr,
                        lambda_fid = args.lambda_fid,
                        lambda_tv = args.lambda_tv,
                        n_blocks = args.n_blocks,
                        n_channels = args.n_channels)
checks.check_model(mode = args.mode,
                   model = model,
                   path = args.model_load_path)


# Train/Test/Run
print('===> Running model...')
output = run(model = model,
             data = data_loaders,
             mode = args.mode,
             epochs = args.epochs,
             val_start = args.val_start,
             val_every = args.val_every,
             autosave_every = args.autosave_every,
             save_path = args.model_save_dir)

print('\n===> Done!')

###