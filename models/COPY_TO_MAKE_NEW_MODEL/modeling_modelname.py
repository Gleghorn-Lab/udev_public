import torch.nn as nn
from transformers import PreTrainedModel


class MODELNAME(PreTrainedModel):
    def __init__(self, config): # takes config
        super().__init__(config) # pass to super
        self.add_your_stuff = config.add_your_variables

    def forward(self,
                input_ids, ### here are some pretty common inputs
                attention_mask=None,
                labels=None,
                output_hidden_states=False,
                output_attentions=False):
        ### implement your forward
        pass
