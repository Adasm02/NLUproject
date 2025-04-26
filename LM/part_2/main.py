# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import *
from utils import collate_fn, load_datasets, store_model_with_params_and_lr

from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import argparse
import numpy as np
import math

#Device selection
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'


# Configurable Parameters
params = {
    'emb_size': 800,
    'hid_size': 800,
    'lr': 5,
    'lr_decay': 0.4,
    'NMT-ASGD' : True,
    'threshold' : 3,
    'clip': 5,
    'emb_dropout': 0.5,
    'out_dropout': 0.5,
    'n_epochs': 30,
    'patience': 5,
    'batch_size': 16,
    'weight_tying' : True,
    'drop_out_mode' : 'variational',
    'optimizer_type': 'SGD',
    'exp_name': 'expLSTM',
    'mode': 'train'
}
emb_size = params['emb_size']
hid_size = params['hid_size']
lr = params['lr']
lr_decay = params['lr_decay']
asgd= params['NMT-ASGD']
threshold = params['threshold']
clip = params['clip']
emb_dropout = params['emb_dropout']
out_dropout = params['out_dropout']
n_epochs = params['n_epochs']
count_patience = params['patience']
batch_size = params['batch_size']
weight_tying = params['weight_tying']
drop_out_mode = params['drop_out_mode']
opt_type = params['optimizer_type']
exp_name = params['exp_name']
mode = params['mode']


if __name__ == "__main__":
    print("running part B"
    "")
    if mode == 'test':
        # Load test dataset
        _, _, test_dataset, lang = load_datasets()
        test_loader = DataLoader(
            test_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device) 
        )
        
        # Load model from checkpoint
        model, criterion_eval = load_checkpoint(
            emb_size, hid_size, lr, clip, device, emb_dropout, out_dropout, lang, opt_type,exp_name
        )
        
        # Evaluate model
        final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test PPL: ', final_ppl)

    else:  # Training mode
        # Load datasets
        train_dataset, dev_dataset, test_dataset, lang = load_datasets()
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"],device=device), shuffle=True
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"],device=device)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"],device=device)
        )

        # Initialize model, optimizer, and loss functions
        model, optimizer, criterion_train, criterion_eval = create_language_model(
            emb_size, hid_size, lr, lr_decay, device, emb_dropout, out_dropout, lang, opt_type, weight_tying, drop_out_mode
        )

        # Training loop
        best_ppl = math.inf
        patience_counter = count_patience
        best_model = None
        losses_train, losses_dev, sampled_epochs, ppl_dev_list = [], [], [], []
        pbar = tqdm(range(1, n_epochs + 1))


        #for NT_ASGD
        best_dev_loss = []
       
        compute_avg = False

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            #scheduler.step() #Reduce the lr of lr_decay percentage after every lr_decay_epoch

            if asgd:
                if compute_avg:
                    tmp = {}
                    for parameter in model.parameters():
                        tmp[parameter] = parameter.data.clone()
                        parameter.data = optimizer.state[parameter]['ax'].clone()

                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                    ppl_dev_list.append(ppl_dev)
                    losses_dev.append(np.asarray(loss_dev).mean())

                    for parameter in model.parameters():
                        parameter.data = tmp[parameter].clone()

                else:

                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                    ppl_dev_list.append(ppl_dev)
                    losses_dev.append(np.asarray(loss_dev).mean())

                    if len(best_dev_loss)>threshold and loss_dev > min(best_dev_loss[:-threshold]):
                        print("\nSwitching optimizer to ASGD")
                        patience = count_patience
                        compute_avg = True
                        params["lr"] *= lr_decay
                        optimizer.param_groups[0]['lr'] = params["lr"]
                        optimizer = optim.ASGD(model.parameters(), lr = params["lr"], t0=0, lambd=0., weight_decay=1.2e-06)
                        

                    best_dev_loss.append(loss_dev)
            else:
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                ppl_dev_list.append(ppl_dev)
                losses_dev.append(np.asarray(loss_dev).mean())


            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = count_patience
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

        # Final evaluation
        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test PPL: ', final_ppl)
        store_model_with_params_and_lr(best_model, optimizer, epoch, exp_name, params, final_ppl)
        plot_losses(losses_train, losses_dev, sampled_epochs, exp_name, save=True)
        plot_ppl(ppl_dev_list, sampled_epochs, exp_name, save=True) 
