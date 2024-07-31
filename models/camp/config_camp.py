from transformers import PretrainedConfig


class CAMPConfig(PretrainedConfig):
    model_type = 'CAMP'
    def __init__(
            self,
            plm_path='facebook/esm2_t6_8m_UR50D',
            nlp_path='allenai/scibert_scivocab_uncased',
            hidden_dim=640,
            intermediate_dim=2560,
            out_dim=512,
            nhead=4, # for convbert
            num_hidden_layers=1, # number convbert layers
            kernel_size=7, # conv kernel
            dropout=0.05, # convbert dropout
            pooling='avg', # pooling at the end
            annotation_transformer=True, # annotations or nlp
            bias=True,
            mnr=False,
            diff=False,
            latent=False,
            mlm=False,
            space=True,
            token=None, # huggingface token, remove before pushing
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.plm_path = plm_path
        self.nlp_path = nlp_path
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.out_dim = out_dim
        self.nhead = nhead
        self.num_hidden_layers = num_hidden_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.pooling = pooling
        self.annotation_transformer = annotation_transformer
        self.mnr = mnr
        self.diff = diff
        self.latent = latent
        self.mlm = mlm
        self.token = token
        self.space = space
        self.bias = bias
