from torch.utils.data import IterableDataset, DataLoader
from torchtext import datasets

def load_dataset(file_path):
    with open(file_path, 'r', encoding= 'utf-8') as f:
        texts = f.read()
        texts = texts.split('\n')
        return texts

class CreateIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

def create_dataset(file_path_en, file_path_de, split):

    en = load_dataset(file_path_en)
    de = load_dataset(file_path_de)
    result = []
    if split == 'train':
        for i in range(0, int(len(en)*0.7)):
            result.append((en[i], de[i]))
    elif split == 'valid':
        for i in range(int(0.7*len(en)), len(en)):
            result.append((en[i], de[i]))
    return CreateIterableDataset(result)
