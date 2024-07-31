from transformers import PreTrainedModel, PretrainedConfig
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein


class ESM3Config(PretrainedConfig):
    model_type = 'esm3'
    def __init__(
        self,
        path = 'esm3_sm_open_v1',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = path


class ESM3Custom(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.path = config.path
        try:
            self.esm = ESM3.from_pretrained(self.path)
        except:
            login()
            self.esm = ESM3.from_pretrained(self.path)
    
    def forward(self, seq):
        tokens = ESMProtein(sequence=seq) # tokenize
        tokens = self.esm.encode(tokens) # to tensor format
        out = self.esm(sequence_tokens = tokens.sequence.unsqueeze(0)) # needs to be batched
        return out
    

if __name__ == '__main__':
    option = input('From ESM or HF (type one): ')
    config = ESM3Config()
    if option.lower() == 'esm':
        model = ESM3Custom(config)
    else:
        model = ESM3Custom.from_pretrained('GleghornLab/esm3', config=config)

    out = model('M')
    print(out)
