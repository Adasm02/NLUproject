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
    'lr':1.5,
    'clip': 5,
    'emb_dropout': 0.2,
    'out_dropout': 0.2,
    'n_epochs': 10,
    'patience': 5,
    'batch_size': 64,
    'optimizer_type': 'SGD',
    'exp_name': 'expLSTM',
    'mode': 'train'
}
emb_size = params['emb_size']
hid_size = params['hid_size']
lr = params['lr']
clip = params['clip']
emb_dropout = params['emb_dropout']
out_dropout = params['out_dropout']
n_epochs = params['n_epochs']
patience = params['patience']
batch_size = params['batch_size']
opt_type = params['optimizer_type']
exp_name = params['exp_name']
mode = params['mode']


if __name__ == "__main__":
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
            emb_size, hid_size, lr, clip, device, emb_dropout, out_dropout, lang, opt_type
        )

        # Training loop
        best_ppl = math.inf
        patience_counter = patience
        best_model = None
        losses_train, losses_dev, sampled_epochs, ppl_dev_list = [], [], [], []
        pbar = tqdm(range(1, n_epochs + 1))

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
            sampled_epochs.append(epoch)
            losses_train.append(np.mean(loss))
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.mean(loss_dev))

            pbar.set_description(f"PPL: {ppl_dev:.4f}")
            
            # Save best model
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience_counter = patience
            else:
                patience_counter -= 1

            # Early stopping
            if patience_counter <= 0:
                break

        # Final evaluation
        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test PPL: ', final_ppl)
        store_model_with_params_and_lr(best_model, optimizer, epoch, exp_name, params, final_ppl)
        plot_losses(losses_train, losses_dev, sampled_epochs, exp_name, save=True)
        plot_ppl(ppl_dev_list, sampled_epochs, exp_name, save=True)
