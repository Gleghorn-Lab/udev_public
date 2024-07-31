from transformers import PretrainedConfig


class MODELNAMEConfig(PretrainedConfig):
    model_type = 'modelname'
    def __init__(
            self,
            ### add necessary variables
            add_your_variables = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_your_variables = add_your_variables
        ### add all to self
