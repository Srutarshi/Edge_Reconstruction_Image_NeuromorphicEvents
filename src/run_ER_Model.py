import os

import numpy as np

import torch


# Train/Test/Run
def run(model, data, mode, epochs, val_start, val_every, autosave_every, save_path):
    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training procedure
    if mode == 'train':
        # Initializations
        best_epoch = 0
        best_loss = 100000

        # First validate (if applicable)
        if data['val']:
            loss = run_epoch(model = model,
                             data = data['val'],
                             mode = 'val',
                             device = device,
                             epoch = 0)
            if loss < best_loss:
                best_loss = loss
                checkpoint(model = model,
                           epoch = 0,
                           save_path = save_path,
                           autosave = False)

        # Training procedure for each batch
        for epoch in range(1, epochs + 1):
            # Train
            model, loss = run_epoch(model = model,
                                    data = data['train'],
                                    mode = 'train',
                                    device = device,
                                    epoch = epoch)

            # Validation procedure
            val_check = (epoch >= val_start) and (epoch % val_every == 0)
            if data['val'] and val_check:
                # Validate
                loss = run_epoch(model = model,
                                 data = data['val'],
                                 mode = 'val',
                                 device = device,
                                 epoch = epoch)

                # Checkpoint saving
                if loss < best_loss:
                    checkpoint(model = model,
                               epoch = epoch,
                               save_path = save_path,
                               autosave = False)
                    best_epoch = epoch
                if epoch % autosave_every == 0:
                    checkpoint(model = model,
                               epoch = epoch,
                               save_path = save_path,
                               autosave = True)
        print('===> Training results...')
        print((' ' * 9) + 'Best epoch: {}'.format(best_epoch))
        print((' ' * 9) + 'Best loss:  {}'.format(best_loss))
        return None

    # Testing procedure
    if mode == 'test':
        loss = run_epoch(model = model,
                         data = data['test'],
                         mode = 'test',
                         device = device)
        return None

    # Running procedure
    if mode == 'run':
        output = run_epoch(model = model,
                           data = data['test'],
                           mode = 'run',
                           device = device)
        return output



# Run one epoch on the data
def run_epoch(model, data, mode, device, epoch = None):
    if mode != 'run':
        total_loss = 0
        n_samples = 0

    if mode == 'train':
        model['network'].train()
    else:
        model['network'].eval()

    for iteration, batch in enumerate(data, 1):
        images = batch['image'].to(device)
        events = batch['event'].to(device)

        if mode == 'train':
            model['optimizer'].zero_grad()

        output = model['network'](images, events)

        if mode != 'run':
            labels = batch['label'].to(device)
            loss = model['loss_fn'](event = events,
                                    pred = output,
                                    label = labels)
            total_loss += loss.item()
            n_samples += len(data)
            avg_loss = total_loss / n_samples

        if mode == 'train':
            loss.backward()
            model['optimizer'].step()
    
    if mode != 'test':
        print((' ' * 9) + 'Avg ' + mode + ' loss at epoch {}: {}'.format(epoch,
                                                                         avg_loss))
    elif mode == 'test':
        print((' ' * 9) + 'Avg ' + mode + ' loss: {}'.format(avg_loss))

    if mode == 'train':
        return model, avg_loss
    elif (mode == 'val') or (mode == 'test'):
        return avg_loss
    elif mode == 'run':
        return output.to(torch.device('cpu'))


# Checkpoint Saving
def checkpoint(model, epoch, save_path, autosave):
    model_name = 'model_'
    message = ''
    if autosave:
        model_name += 'epoch_{}.pth'.format(epoch)
        message = 'Checkpoint saved at epoch {}'.format(epoch)
    else:
        model_name += 'best.pth'
        message = 'Best model saved at epoch {}'.format(epoch)

    model_out_path = os.path.join(save_path,
                                  model_name)
    torch.save(model['network'].state_dict(), model_out_path)
    print((' ' * 13) + message)

###