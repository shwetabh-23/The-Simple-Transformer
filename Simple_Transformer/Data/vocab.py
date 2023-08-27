import os
from .dataset import load_dataset
import spacy
from collections import Counter

UNK_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3

UNK = '<UNK>'
PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
FS = '.'

special_characters = [UNK, PAD, SOS, EOS, FS]

class Vocab:
    def __init__(self, tokenizer, tokens):
        self.tokenizer = tokenizer
        self.tokens = tokens + special_characters
        self.index_lookup = {
            self.tokens[i] : i for i in range(len(self.tokens))
        }

    def __len__(self):
        return len(self.tokens)

    def __call__(self, text):
        text = text.strip()
        return [self.to_idx(token.text) for token in self.tokenizer(text)]


    def to_idx(self, token):
        return self.index_lookup[token] if token in self.tokens else self.index_lookup[UNK]

    def tokenize(self, text):
        text = text.strip()
        return[token.text for token in self.tokenizer(text) if len(self.token.text.strip()) > 0]


def generate_tokens(tokenizer, texts):
    texts = [text.strip() for text in texts]
    tokens = []
    for doc in tokenizer.pipe(texts):
        for req_token in ([token.text for token in doc]):
            tokens.append(req_token)
    return tokens

def save_tokens(path, tokens):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, 'w', encoding= 'utf-8') as f:
        f.writelines(' '.join(tokens)) 

def load_tokens(path):
    tokens = []  # Initialize an empty list to store tokens
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()  # Read the entire text from the file
        sentences = text.split('.')  # Split text into sentences using full stops
        for sentence in sentences:
            words = sentence.strip().split()  # Split sentence into words
            tokens.extend(words)  # Append words to the tokens list
    return tokens

def load_vocab(path_to_tokens, texts, language):
    tokenizer = spacy.load(language)
    try:
        tokens = load_tokens(path_to_tokens)
    except:
        tokens = []
        pass
    if len(tokens) == 0:
        print("Tokens not found, creating new tokens")
        tokens = generate_tokens(tokenizer=tokenizer, texts= texts)
        save_tokens(path_to_tokens, tokens)
        print("Process finished, created new tokens")

    return Vocab(tokenizer, tokens)

def load_vocab_pair(en_path, ger_path):
    eng_texts = load_dataset(en_path)
    german_texts = load_dataset(ger_path)

    current_dir = os.getcwd()
    path_to_eng_tokens = os.path.join(current_dir, 'eng_tokens.txt')
    path_to_ger_tokens = os.path.join(current_dir, 'ger_tokens.txt')

    eng_vocab = load_vocab(path_to_eng_tokens, eng_texts, 'en_core_web_sm')
    german_vocab = load_vocab(path_to_ger_tokens, german_texts, 'de_core_news_sm')
    
    return eng_vocab, german_vocab
