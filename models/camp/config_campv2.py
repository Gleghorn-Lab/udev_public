from transformers import PretrainedConfig


class CAMPv2Config(PretrainedConfig):
    model_type = 'CAMPv2'
    def __init__(
            self,
            context_path='facebook/esm2_t6_8m_UR50D', # context encoder path
            target_path='facebook/esm2_t6_8m_UR50D',  # target encoder path
            annotation_path='GleghornLab/AT_RED',     # annotation encoder path
            contrastive_loss='space',                 # space or mnr
            common_dim=512,     # common projected dim
            context=True,       # use context encoder
            target=True,        # use target encoder
            at_ce_hyper=1.0,    # annotation cross entropy hyperparameter
            tg_ce_hyper=0.1,    # target cross entropy hyperparameter
            tg_cont_hyper=10.0, # target contrastive loss hyperparameter
            ct_ce_hyper=5.0,    # context cross entropy hyperparameter
            l1_hyper=1.0,       # l1 loss hyperparameter
            space_lambda1=1.0,    # space loss primary hyper
            space_lambda2=1.0,     # space loss secondary hyper
            token=None,         # huggingface token, remove before pushing
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_path = context_path
        self.target_path = target_path
        self.annotation_path = annotation_path
        self.contrastive_loss = contrastive_loss
        self.common_dim = common_dim
        self.context = context
        self.target = target
        self.token = token
        self.at_ce_hyper = at_ce_hyper
        self.tg_ce_hyper = tg_ce_hyper
        self.tg_cont_hyper = tg_cont_hyper
        self.ct_ce_hyper = ct_ce_hyper
        self.l1_hyper = l1_hyper
        self.space_lambda1 = space_lambda1
        self.space_lambda2 = space_lambda2
