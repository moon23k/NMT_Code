import os, re, json, yaml
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




#Conala
def process_conala_data(data_volumn=101100):
    volumn_cnt = 0
    min_len, max_len = 10, 300
    corpus, processed_data = [], []

    data = load_dataset("neulab/conala", "mined")['train']
    for elem in data:
        x = elem['intent'].lower()
        y = elem['snippet']

        x_len, y_len = len(x), len(y)
        min_condition = (x_len >= min_len) & (y_len >= min_len)
        max_condition = (x_len <= max_len) & (y_len <= max_len)

        if max_condition & min_condition:
            corpus.append(x)
            corpus.append(y)
            processed_data.append({'x': x, 'y': y})

            volumn_cnt += 1
        
        if volumn_cnt == data_volumn:
            break

    with open('data/corpus.txt', 'w') as f:
        json.dump(corpus, f)

    return processed_data



def train_tokenizer():
    corpus_path = 'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(BPE(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_config['vocab_size'], 
        special_tokens=[
            vocab_config['pad_token'], 
            vocab_config['unk_token'],
            vocab_config['bos_token'],
            vocab_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save("data/tokenizer.json")



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-1100], data_obj[-1100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')




def main():
    #Process Data
    processed = process_conala_data()

    #Train Tokenizer
    train_tokenizer()

    #Save Data
    save_data(processed)


if __name__ == '__main__':
    main()    