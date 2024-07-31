from transformers import PretrainedConfig


class ConvBertConfig(PretrainedConfig):
    model_type = 'convbert'
    def __init__(
            self,
            input_size = 512, # for probe
            hidden_size = 768,
            nhead = 4,
            intermediate_size = 3072,
            num_layers = 1,
            kernel_size = 7,
            dropout = 0.1,
            pooling = 'max',
            task_type = 'singlelabel',
            num_labels = 2,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.pooling = pooling
        self.task_type = task_type
        self.num_labels = num_labels
