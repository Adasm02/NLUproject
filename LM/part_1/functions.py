# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math
import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMLanguageModel, RNNLanguageModel
import os
import matplotlib.pyplot as plt

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def create_language_model(embedding_dim, hidden_dim, learning_rate, clip, device, emb_dropout, output_dropout, lang, opt_type):
    """
    Creates and initializes the language model.
    """
    vocab_size = len(lang.word2id)
    model = LSTMLanguageModel(embedding_dim, hidden_dim, vocab_size, pad_index=lang.word2id["<pad>"], out_dropout=output_dropout, emb_dropout=emb_dropout).to(device)
    model.apply(init_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) if opt_type == 'SGD' else optim.AdamW(model.parameters(), lr=learning_rate)
    train_criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    eval_criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    print(model)
    return model, optimizer, train_criterion, eval_criterion




def load_checkpoint(
            embedding_dim, hidden_dim, lr, clip, device, emb_dropout, out_dropout, lang, optimizer_type, exp_name
):
    """
    Loads a model checkpoint from disk.
    """
    
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", exp_name, "best_model.pt")
    
    vocab_size = len(lang.word_to_id) 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Choose model class based on experiment name
    model_class = RNNLanguageModel if exp_name == "expRNN" else LSTMLanguageModel
    
    # Initialize the model
    model = model_class(
        embedding_dim=embedding_dim, 
        hidden_dim=hidden_dim, 
        vocab_size=vocab_size, 
        pad_index=lang.word2id["<pad>"]
    ).to(device)
    
    # Load the model's state dict
    model.load_state_dict(checkpoint)
    
    # Define the evaluation criterion
    eval_criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    return model, eval_criterion

def plot_losses(losses_train, losses_dev, sampled_epochs, exp_name, save=False):
    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o')
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss', marker='s')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        plot_dir = os.path.join(project_dir, 'plot', exp_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'losses.png')
        plt.savefig(plot_path)
        print(f"Loss plot saved at {plot_path}")
    
    # plt.show() è disattivato per evitare warning in ambienti non interattivi
    # plt.show()

def plot_ppl(ppl_dev_list, sampled_epochs, exp_name, save=False):
    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, ppl_dev_list, label='Validation PPL', marker='o')

    plt.title('Validation PPL')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        plot_dir = os.path.join(project_dir, 'plot', exp_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'ppl.png')
        plt.savefig(plot_path)
        print(f"PPL plot saved at {plot_path}")
    
    # plt.show() è disattivato per evitare warning in ambienti non interattivi
    # plt.show()

