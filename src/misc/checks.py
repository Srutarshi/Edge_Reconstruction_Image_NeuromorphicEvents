import os

import torch


# Check all args
def check_args(args):
    assert_at_least(arg = '--autosave_every',
                    argval = args.autosave_every,
                    val = 1)
    assert_at_least(arg = '--batch_size',
                    argval = args.batch_size,
                    val = 1)
    assert_at_least(arg = '--epochs',
                    argval = args.epochs,
                    val = 1)
    assert_at_least(arg = '--lambda_fid',
                    argval = args.lambda_fid,
                    val = 0)
    assert_at_least(arg = '--lambda_tv',
                    argval = args.lambda_tv,
                    val = 0)
    assert_at_least(arg = '--n_blocks',
                    argval = args.n_blocks,
                    val = 1)
    assert_at_least(arg = '--n_channels',
                    argval = args.n_channels,
                    val = 1)
    assert_at_least(arg = '--threads',
                    argval = args.threads,
                    val = 1)
    assert_at_least(arg = '--thresh',
                    argval = args.thresh,
                    val = 0)
    assert_at_least(arg = '--val_every',
                    argval = args.val_every,
                    val = 1)
    assert_at_least(arg = '--val_start',
                    argval = args.val_start,
                    val = 0)

    assert_mode(mode = args.mode)

    if args.mode == 'train':
        assert_valid_path(arg = '--train_files',
                          path = args.train_files)
        assert_valid_path(arg = '--val_files',
                          path = args.val_files)
    if args.mode == 'train' or args.mode == 'test':
        assert_valid_path(arg = '--test_files',
                          path = args.test_files)
    if args.model_load_path:
        assert_valid_path(arg = '--model_load_path',
                          path = args.model_load_path)

    assert_model(path = args.model_save_dir,
                 mode = args.mode)

    if args.shuffle_data:
        assert_split(split_name = '--split_train',
                     split = args.split_train)
        assert_split(split_name = '--split_test',
                     split = args.split_test)
        assert_split_sum(split_train = args.split_train,
                         split_test = args.split_test)

    print((' ' * 9) + 'Arguments accepted')


# Check data/datasets
def check_data(data, mode, check):
    space = ' ' * 9
    err_app = ''
    prep_app = ' prepared'
    no_app = ''
    accept = 'Data accepted'
    if check == 'datasets':
        err_app = ' was located'
        prep_app = 'set built'
        no_app = 'set'
        accept = 'Datasets accepted'
    if check == 'dataloaders':
        err_app = 'loader created'
        prep_app = ' loader created'
        no_app = ' loader'
        accept = 'Data loaders accepted'

    if mode == 'train':
        cond = len(data['train']) > 0
        err = 'No training data' + err_app
        assert cond, err

        cond = len(data['val']) > 0
        err = 'No validation data' + err_app
        assert cond, err
    elif mode == 'test':
        cond = len(data['test']) > 0
        err = 'No testing data' + err_app
        assert cond, err

    if data['train']:
        message = 'Training data' + prep_app
        print(space + message)
    else:
        message = 'No training data' + no_app
        print(space + message)

    if data['test']:
        message = 'Testing data' + prep_app
        print(space + message)
    else:
        message = 'No testing data' + no_app
        print(space + message)

    if data['val']:
        message = 'Validation data' + prep_app
        print(space + message)
    else:
        message = 'No validation data' + no_app
        print(space + message)

    print(space + accept)


# Check model
def check_model(mode, model, path):
    space = ' ' * 9

    cond = (model['network'] is not None)
    err = 'Network was not initialized properly'
    assert cond, err
    message = 'Network initialized '
    if path:
        message += ('with weights from ' + path)
    else:
        message += 'at random'
    print(space + message)

    if mode == 'train':
        cond = model['loss_fn'] is not None
        err = 'Loss function was not initialized properly'
        assert cond, err
        print(space + 'Loss function initialized')

        cond = model['optimizer'] is not None
        err = 'Optimizer was not initialized properly'
        assert cond, err
        print(space + 'Optimizer initialized')

    print(space + 'Model accepted')


# Assert an argument is at least some number
def assert_at_least(arg, argval, val):
    cond = argval >= val
    err = '{} was assigned {}, but should be at least {}'.format(arg,
                                                                            argval,
                                                                            val)
    assert cond, err

# Assert CUDA is available if desired
def assert_cuda(cuda):
    cond = (cuda and torch.cuda.is_available()) or (not cuda)
    err = '--cuda was assigned 1, but no GPU was found'
    assert cond, err

# Assert mode is train or test
def assert_mode(mode):
    mode = mode.lower()
    cond = (mode == 'train') or (mode == 'test')
    err = '--mode was assigned ' + mode + ', but should be \'train\' or \'test\''
    assert cond, err

# Assert/Confirm model
def assert_model(path, mode):
    if mode.lower() == 'train' and os.path.exists(path):
        print('     (!) WARNING: Contents in ' + path + ' will be overwritten')
    elif mode.lower() == 'train' and not os.path.exists(path):
        os.makedirs(path)
    elif mode.lower() == 'test':
        cond = os.path.exists(path)
        err = '--model_dir ' + path + ' does not exist'
        assert cond, err

# Assert if train/val/test split is within bounds
def assert_split(split_name, split):
    cond = (split >= 0 and split <= 1)
    err = '{} was assigned {}, but should be [0,1]'.format(split_name,
                                                           split)
    assert cond, err

# Assert the sum of the splits is at most 1
def assert_split_sum(split_train, split_test):
    split_sum = split_train + split_test
    cond = (split_sum <= 1)
    err = '--split_train ({}) and --split_test ({}) have sum {} > 1'.format(split_train,
                                                                            split_test,
                                                                            split_sum)
    assert cond, err

# Assert path is valid
def assert_valid_path(arg, path):
    cond = os.path.exists(path)
    err = path + ' is not an existing path for ' + arg
    assert cond, err

###