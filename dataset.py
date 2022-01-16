from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

PAD = '[PAD]'
UNK = '[UNK]'

class BehaviorDataset(Dataset):
    def __init__(self, data_path, vocab_path, pad_size=256):
        self.vocab_dict = self.build_vocab(vocab_path)
        self.pad_size = pad_size
        self.contents = self.build_dataset(data_path)

    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, index):
        words, label = self.contents[index]
        return np.array(words, dtype=np.long), label

    def build_vocab(self, vocab_path):
        word_list = []
        with open(vocab_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                word = line.strip()
                word_list.append(word)

        vocab_dict = {word: idx for idx, word in enumerate(word_list)}
        return vocab_dict

    def build_dataset(self, data_path):
        contents = []
        tokenizer = lambda x: x.split(' ')
        with open(data_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                l = line.strip()
                content, label = l.split(',')
                words = []
                token = tokenizer(content)

                if len(token) < self.pad_size:
                    token.extend([PAD] * (self.pad_size - len(token)))
                else:
                    token = token[:self.pad_size]

                for word in token:
                    words.append(self.vocab_dict.get(word, self.vocab_dict.get(UNK)))
                
                contents.append((words, int(label)))
        
        return contents
        

