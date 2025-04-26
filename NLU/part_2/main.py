# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
import copy

#Device selection
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA available")
    torch.cuda.empty_cache()
else:
    device = 'cpu'

PAD_TOKEN = 0

# Configurable Parameters
params = {
    'hid_size': 768,
    'lr': 5e-5,
    'clip':5,
    'n_epochs': 20,
    'patience': 5,
    'batch_size': 32,
    'dropout': 0.1,
    'portion' : 0.10
}
hid_size = params['hid_size']
lr = params['lr']
clip = params['clip']
n_epochs = params['n_epochs']
count_patience = params['patience']
batch_size = params['batch_size']
dropout = params['dropout']
portion = params['portion']

base_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(os.path.dirname(base_path), "dataset")

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    tmp_train_raw = load_data(os.path.join(data_path,'ATIS','train.json'))
    test_raw = load_data(os.path.join(data_path,'ATIS','test.json'))
    labels = []
    inputs = []
    mini_train = []
    intent_acc_vector,slot_f1s_vector = [],[]
    intents = [x['intent'] for x in tmp_train_raw] 
    count_y = Counter(intents)

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: 
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Create the train and dev set
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]
    words = sum([x['utterance'].split() for x in train_raw], []) 

    corpus = train_raw + dev_raw + test_raw 
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    lang = Lang(intents, slots, cutoff=0)


    # create the datasets
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    

    model = ModelBert(hid_size, out_slot, out_int, dropout).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token


    patience = count_patience
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    best_score = 0
    for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model, clip=clip)
            if x % 1 == 0: 
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang,tokenizer)
                losses_dev.append(np.asarray(loss_dev).mean())
                
                f1 = results_dev['total']['f']
                intent_acc = intent_res["accuracy"]
                actual_score = (f1 + intent_acc) / 2
                if actual_score > best_score:
                    best_score = actual_score
                    # Save the best model
                    best_model = copy.deepcopy(model).to('cpu')
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patience
                    break 

                best_model.to(device)
                results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang, tokenizer)    
                test_f1 = results_test['total']['f']
                test_acc = intent_test['accuracy']
                intent_acc_vector.append(intent_test['accuracy'])
                slot_f1s_vector.append(results_test['total']['f'])



    slot_f1s_vector = np.asarray(slot_f1s_vector)
    intent_acc_vector = np.asarray(intent_acc_vector)
    print('Slot F1', round(slot_f1s_vector.mean(),3), '+-', round(slot_f1s_vector.std(),3))
    print('Intent Acc', round(intent_acc_vector.mean(), 3), '+-', round(intent_acc_vector.std(), 3))
    store_model_with_params_and_lr(params,slot_f1s_vector.mean(),slot_f1s_vector.std(), intent_acc_vector.mean(), intent_acc_vector.std() )



