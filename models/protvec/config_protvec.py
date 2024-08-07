from transformers import PretrainedConfig


class ProteinVecConfig(PretrainedConfig):
    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model",
                     "num_attention_heads": "num_heads",
                     "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        ### T5
        vocab_size=128,
        d_model=1024,
        d_kv=128,
        d_ff=16384,
        num_layers=24,
        num_decoder_layers=None,
        num_heads=32,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=None,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        classifier_dropout=0.0,

        ### Aspect Vecs
        ec_d_model=1024,
        ec_nhead=4,
        ec_num_layers=2,
        ec_dim_feedforward=2048,
        ec_out_dim=512,
        ec_dropout=0.1,
        ec_activation="relu",
        ec_num_variables=10,
        ec_vocab=20,
        ec_lr0=0.0001,
        ec_warmup_steps=500,
        ec_p_bernoulli=0.5,

        gene3d_d_model=1024,
        gene3d_nhead=4,
        gene3d_num_layers=2,
        gene3d_dim_feedforward=2048,
        gene3d_out_dim=512,
        gene3d_dropout=0.1,
        gene3d_activation="relu",
        gene3d_num_variables=10,
        gene3d_vocab=20,
        gene3d_lr0=0.0001,
        gene3d_warmup_steps=500,
        gene3d_p_bernoulli=0.5,

        bp_d_model=1024,
        bp_nhead=4,
        bp_num_layers=4,
        bp_dim_feedforward=2048,
        bp_out_dim=512,
        bp_dropout=0.1,
        bp_activation="relu",
        bp_num_variables=10,
        bp_vocab=20,
        bp_lr0=0.0001,
        bp_warmup_steps=500,
        bp_p_bernoulli=0.5,

        cc_d_model=1024,
        cc_nhead=4,
        cc_num_layers=4,
        cc_dim_feedforward=2048,
        cc_out_dim=512,
        cc_dropout=0.1,
        cc_activation="relu",
        cc_num_variables=10,
        cc_vocab=20,
        cc_lr0=0.0001,
        cc_warmup_steps=500,
        cc_p_bernoulli=0.5,

        mf_d_model=1024,
        mf_nhead=4,
        mf_num_layers=4,
        mf_dim_feedforward=2048,
        mf_out_dim=512,
        mf_dropout=0.1,
        mf_activation="relu",
        mf_num_variables=10,
        mf_vocab=20,
        mf_lr0=0.0001,
        mf_warmup_steps=500,
        mf_p_bernoulli=0.5,

        pfam_d_model=1024,
        pfam_nhead=4,
        pfam_num_layers=2,
        pfam_dim_feedforward=2048,
        pfam_out_dim=512,
        pfam_dropout=0.1,
        pfam_activation="relu",
        pfam_num_variables=10,
        pfam_vocab=20,
        pfam_lr0=0.0001,
        pfam_warmup_steps=500,
        pfam_p_bernoulli=0.5,

        tm_d_model=1024,
        tm_nhead=4,
        tm_num_layers=4,
        tm_dim_feedforward=2048,
        tm_out_dim=512,
        tm_dropout=0.1,
        tm_activation="relu",
        tm_lr0=0.0001,
        tm_warmup_steps=300,

        vec_d_model=512,
        vec_nhead=4,
        vec_num_layers=2,
        vec_dim_feedforward=2048,
        vec_out_dim=512,
        vec_dropout=0.1,
        vec_activation="relu",
        vec_num_variables=10,
        vec_vocab=20,
        vec_lr0=0.0001,
        vec_warmup_steps=500,
        vec_p_bernoulli=0.5,
        
        inference_aspect = 'ALL',
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        # EC parameters
        self.ec_d_model = ec_d_model
        self.ec_nhead = ec_nhead
        self.ec_num_layers = ec_num_layers
        self.ec_dim_feedforward = ec_dim_feedforward
        self.ec_out_dim = ec_out_dim
        self.ec_dropout = ec_dropout
        self.ec_activation = ec_activation
        self.ec_num_variables = ec_num_variables
        self.ec_vocab = ec_vocab
        self.ec_lr0 = ec_lr0
        self.ec_warmup_steps = ec_warmup_steps
        self.ec_p_bernoulli = ec_p_bernoulli

        # GENE3D parameters
        self.gene3d_d_model = gene3d_d_model
        self.gene3d_nhead = gene3d_nhead
        self.gene3d_num_layers = gene3d_num_layers
        self.gene3d_dim_feedforward = gene3d_dim_feedforward
        self.gene3d_out_dim = gene3d_out_dim
        self.gene3d_dropout = gene3d_dropout
        self.gene3d_activation = gene3d_activation
        self.gene3d_num_variables = gene3d_num_variables
        self.gene3d_vocab = gene3d_vocab
        self.gene3d_lr0 = gene3d_lr0
        self.gene3d_warmup_steps = gene3d_warmup_steps
        self.gene3d_p_bernoulli = gene3d_p_bernoulli

        # BP parameters
        self.bp_d_model = bp_d_model
        self.bp_nhead = bp_nhead
        self.bp_num_layers = bp_num_layers
        self.bp_dim_feedforward = bp_dim_feedforward
        self.bp_out_dim = bp_out_dim
        self.bp_dropout = bp_dropout
        self.bp_activation = bp_activation
        self.bp_num_variables = bp_num_variables
        self.bp_vocab = bp_vocab
        self.bp_lr0 = bp_lr0
        self.bp_warmup_steps = bp_warmup_steps
        self.bp_p_bernoulli = bp_p_bernoulli

        # CC parameters
        self.cc_d_model = cc_d_model
        self.cc_nhead = cc_nhead
        self.cc_num_layers = cc_num_layers
        self.cc_dim_feedforward = cc_dim_feedforward
        self.cc_out_dim = cc_out_dim
        self.cc_dropout = cc_dropout
        self.cc_activation = cc_activation
        self.cc_num_variables = cc_num_variables
        self.cc_vocab = cc_vocab
        self.cc_lr0 = cc_lr0
        self.cc_warmup_steps = cc_warmup_steps
        self.cc_p_bernoulli = cc_p_bernoulli

        # MF parameters
        self.mf_d_model = mf_d_model
        self.mf_nhead = mf_nhead
        self.mf_num_layers = mf_num_layers
        self.mf_dim_feedforward = mf_dim_feedforward
        self.mf_out_dim = mf_out_dim
        self.mf_dropout = mf_dropout
        self.mf_activation = mf_activation
        self.mf_num_variables = mf_num_variables
        self.mf_vocab = mf_vocab
        self.mf_lr0 = mf_lr0
        self.mf_warmup_steps = mf_warmup_steps
        self.mf_p_bernoulli = mf_p_bernoulli

        # PFAM parameters
        self.pfam_d_model = pfam_d_model
        self.pfam_nhead = pfam_nhead
        self.pfam_num_layers = pfam_num_layers
        self.pfam_dim_feedforward = pfam_dim_feedforward
        self.pfam_out_dim = pfam_out_dim
        self.pfam_dropout = pfam_dropout
        self.pfam_activation = pfam_activation
        self.pfam_num_variables = pfam_num_variables
        self.pfam_vocab = pfam_vocab
        self.pfam_lr0 = pfam_lr0
        self.pfam_warmup_steps = pfam_warmup_steps
        self.pfam_p_bernoulli = pfam_p_bernoulli

        # Vec parameters
        self.vec_d_model = vec_d_model
        self.vec_nhead = vec_nhead
        self.vec_num_layers = vec_num_layers
        self.vec_dim_feedforward = vec_dim_feedforward
        self.vec_out_dim = vec_out_dim
        self.vec_dropout = vec_dropout
        self.vec_activation = vec_activation
        self.vec_num_variables = vec_num_variables
        self.vec_vocab = vec_vocab
        self.vec_lr0 = vec_lr0
        self.vec_warmup_steps = vec_warmup_steps
        self.vec_p_bernoulli = vec_p_bernoulli

        # TM parameters
        self.tm_d_model = tm_d_model
        self.tm_nhead = tm_nhead
        self.tm_num_layers = tm_num_layers
        self.tm_dim_feedforward = tm_dim_feedforward
        self.tm_out_dim = tm_out_dim
        self.tm_dropout = tm_dropout
        self.tm_activation = tm_activation
        self.tm_lr0 = tm_lr0
        self.tm_warmup_steps = tm_warmup_steps

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        self.inference_aspect = inference_aspect

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

