# Add functions or classes used for data loading and preprocessing
import torch
import torch.utils.data as data
import os
import json

def read_file(path, eos_token="<eos>"):
    """
    Reads a text file line by line, removes any leading and trailing spaces, 
    and appends an end-of-sequence token to each line.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    """
    Mapping words to unique ids
    """
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output
def collate_fn(data, pad_token, device):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item


def load_datasets():
    """
    Function for loading and preprocessing dataset 
    """
    base_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(os.path.dirname(base_path), "part_B", "dataset","PennTreeBank" )
    
    train_text = read_file(os.path.join(data_path, "ptb.train.txt"))
    val_text = read_file(os.path.join(data_path, "ptb.valid.txt"))
    test_text = read_file(os.path.join(data_path, "ptb.test.txt"))
    
    vocab = Lang(train_text, ["<pad>", "<eos>"])
    train_set = PennTreeBank(train_text, vocab)
    val_set = PennTreeBank(val_text, vocab)
    test_set = PennTreeBank(test_text, vocab)
    
    return train_set, val_set, test_set, vocab

def store_model_with_params_and_lr(model, optimizer, epoch, experiment_name, params, final_ppl):
    """
    Function to save model along with learning rate, other details, and a txt file with model parameters.
    """
    
    project_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(project_dir, 'bin', experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model_with_lr.pt')
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'learning_rate': optimizer.param_groups[0]['lr']
    }

    torch.save(checkpoint, model_path)
    print("Model and optimizer state saved at", model_path)

    params_path = os.path.join(model_dir, 'model_params.txt')
    
    with open(params_path, 'w') as f:
        f.write("Model parameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"PPL on test: {final_ppl}\n")

    print("Model parameters saved at", params_path)


# Class for handling vocabulary mappings
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

# Dataset class for handling tokenized sentences
class PennTreeBank(data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        for sentence in corpus:
            self.source.append(sentence.split()[:-1])
            self.target.append(sentence.split()[1:])
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') 
                    break
            res.append(tmp_seq)
        return res

