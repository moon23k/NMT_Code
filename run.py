import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from transformers import (
    set_seed, 
    RobertaTokenizer, 
    T5ForConditionalGeneration
)

from module import (
    load_dataloader,
    load_model,
    Trainer, 
    Tester
)



class Config(object):
    def __init__(self, args):    

        #Get Config Attributes from config.yaml file
        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.mode = args.mode
        self.search_method = args.search
        self.model_type = args.model_type
        self.mname = 'Salesforce/codet5-base'

        self.tokenizer_path = "data/tokenizer.json"
        self.ckpt = f"ckpt/{self.model_type}_model.pt"

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer


def inference():
    return


def main(config):
    set_seed(42)
    config = Config(args)

    tokenizer = RobertaTokenizer.from_pretrained(config.mname)
    model = T5ForConditionalGeneration.from_pretrained(config.mname, model_max_length=config.max_len)

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')        
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()        
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
        
    elif config.mode == 'inference':
        inference()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    
    main(args)