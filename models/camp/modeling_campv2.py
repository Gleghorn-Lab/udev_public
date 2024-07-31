import torch
import torch.nn as nn
from transformers import PreTrainedModel, EsmTokenizer
from functools import partial
from ..outputs import GeneralModelOutput
from ...losses.contrastive import space_loss, MNR_loss
from ..esm.custom_esm import CustomEsmForMaskedLM


class CAMPv2(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = EsmTokenizer.from_pretrained(config.context_path, token=config.token)
        common_dim = config.common_dim # c
        if config.contrastive_loss == 'mnr':
            self.contrastive_loss = MNR_loss
        else:
            self.contrastive_loss = partial(space_loss, lambda_1=config.space_lambda1, lambda_2=config.space_lambda2)
            
        self.annotation_transformer = CustomEsmForMaskedLM.from_pretrained(config.annotation_path, token=config.token) # d3
        self.annotation_pool_proj = nn.Linear(self.annotation_transformer.config.hidden_size, common_dim)

        self.target = config.target
        if config.target:
            self.target_encoder = CustomEsmForMaskedLM.from_pretrained(
                config.target_path,
                token=config.token,
                token_dropout=False,
                emb_layer_norm_before=True,
                hidden_act='silu',
                max_position_embeddings=100,
                ignore_mismatched_sizes=True
            ) # d2
            self.target_state_proj = nn.Linear(self.target_encoder.config.hidden_size, common_dim)
            self.target_pool_proj = nn.Linear(self.target_encoder.config.hidden_size, common_dim)

        self.context = config.context
        if config.context:
            self.context_encoder = CustomEsmForMaskedLM.from_pretrained(
                config.context_path,
                is_decoder=True,
                add_cross_attention=True,
                emb_layer_norm_before=True,
                hidden_act='silu',
                position_embedding_type="absolute",
                max_position_embeddings=2048,
                token_dropout=False,
                token=config.token,
                ignore_mismatched_sizes=True
            ) # d1
            self.context_state_proj = nn.Linear(self.context_encoder.config.hidden_size, common_dim)
            self.context_pool_proj = nn.Linear(self.context_encoder.config.hidden_size, common_dim)
            self.annotation_state_proj = nn.Linear(self.annotation_transformer.config.hidden_size, self.context_encoder.config.hidden_size)
            self.l1 = nn.SmoothL1Loss()

        self.at_ce_hyper = config.at_ce_hyper
        self.tg_ce_hyper = config.tg_ce_hyper
        self.tg_cont_hyper = config.tg_cont_hyper
        self.ct_ce_hyper = config.ct_ce_hyper
        self.l1_hyper = config.l1_hyper
        self.special_token_ids = [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.mask_token_id]
        self.ce = nn.CrossEntropyLoss()

    def _mask_tokens(self, inputs, labels, prob=0.30):
        probability_matrix = torch.full(inputs.shape, prob)
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_tokens_mask |= (inputs == token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # Ensure at least one token is masked
        if not masked_indices.any():
            # Find indices of non-special tokens
            non_special_indices = (~special_tokens_mask).nonzero(as_tuple=True)[1]
            if len(non_special_indices) > 0:
                # Randomly choose one non-special token to mask
                random_index = non_special_indices[torch.randint(0, len(non_special_indices), (1,))]
                masked_indices[0, random_index] = True
        labels[~masked_indices] = -100  # Set labels for non-masked tokens to -100
        inputs[masked_indices] = self.tokenizer.mask_token_id  # Replace masked tokens with mask token ID
        return inputs, labels

    def sort_and_update(self, sorted_probs, k, ids):
        # sorted_probs, dictionary - int: (float, int) - index: (probability, predicted_token)
        # k, int - number of items to keep
        # ids, tensor - replace with predicted tokens
        # Sort the dictionary by the probabilities
        sorted_items = sorted(sorted_probs.items(), key=lambda x: x[1][0], reverse=True)
        # Take only up to k items, but don't exceed the number of masked tokens
        top_k_items = sorted_items[:min(k, len(sorted_items))]
        for idx, (prob, token) in top_k_items:
            ids[idx] = token
        return ids    

    def generate(
        self,
        annotation_input_ids,
        annotation_attention_mask,
        sequence=None,
        prob=0.30,
        seq_len=None,
        k=1,
        entropy=False,
        device='cpu',
        view=False,
    ):
        annotation_input_ids, annotation_attention_mask = annotation_input_ids.to(device), annotation_attention_mask.to(device)
        if sequence != None:
            seq_len = len(sequence)
            template = self.tokenizer(sequence, add_special_tokens=True, return_tensors='pt')
            template_ids, template_attention_mask = template['input_ids'], template['attention_mask']
            labels = template_ids.clone()
            template_ids, labels = self._mask_tokens(template_ids, labels, prob=prob)
            template_ids, template_attention_mask, labels = template_ids.to(device), template_attention_mask.to(device), labels.to(device)
        else:
            labels = None
            template = self.tokenizer('M', add_special_tokens=False)
            template_ids = template['input_ids']
            template_ids = [self.tokenizer.cls_token_id] + template_ids + [self.tokenizer.mask_token_id] * (seq_len-1) + [self.tokenizer.eos_token_id]
            template_ids = torch.tensor(template_ids).unsqueeze(0).to(device)
            template_attention_mask = torch.ones_like(template_ids, device=device)

        at_out = self.annotation_transformer(
            input_ids=annotation_input_ids,
            attention_mask=annotation_attention_mask
        )
        at_state = at_out.last_hidden_state # (B, L, d3)
        at_state = self.annotation_state_proj(at_state) # (B, L, d1)

        if k > seq_len:
            logits = self.context_encoder(
                input_ids=template_ids,
                attention_mask=template_attention_mask,
                encoder_hidden_states=at_state,
                encoder_attention_mask=annotation_attention_mask,
                output_full_logits=True,
            ).logits
            print(logits.shape)
        else:
            for _ in range(seq_len):
                current_seq = self.tokenizer.decode(template_ids[0]).replace(' ', '').replace('<mask>', '-').replace('<cls>', '').replace('<eos>', '')
                if view:
                    print(current_seq)
                mask_indices = (template_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                if len(mask_indices) == 0:
                    break  # No more masks to fill
                logits = self.context_encoder(
                    input_ids=template_ids,
                    attention_mask=template_attention_mask,
                    encoder_hidden_states=at_state,
                    encoder_attention_mask=annotation_attention_mask,
                    output_full_logits=True,
                ).logits
                if entropy:
                    probs = logits.softmax(dim=-1).squeeze(0)
                else:
                    probs = logits.squeeze(0)
                sorted_probs = {}
                for idx in mask_indices:
                    mask_fill_prob = probs[idx]
                    sorted_probs[idx.item()] = (mask_fill_prob.max().item(), mask_fill_prob.argmax(dim=-1).item())
                template_ids = self.sort_and_update(sorted_probs, k, template_ids.squeeze(0))
                template_ids = template_ids.unsqueeze(0)
        
        if labels is not None:
            loss = self.ce(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
            print('Loss: ', loss.item())
        return current_seq, logits

    def forward(
        self,
        target_input_ids,
        target_attention_mask,
        annotation_input_ids,
        annotation_attention_mask,
        target_labels=None,
        annotation_labels=None,
        context_input_ids=None,
        context_attention_mask=None,
        context_labels=None,
        output_attentions=False,
        labels=None # so compute metrics works
    ):
        logits, losses = [], []
        at_out = self.annotation_transformer(
            input_ids=annotation_input_ids,
            attention_mask=annotation_attention_mask,
            labels=annotation_labels,
            output_full_logits=False
        )
        at_state = at_out.last_hidden_state # (B, L, d3)
        at_pooled = at_out.pooler_output # (B, d3)
        at_pooled = self.annotation_pool_proj(at_pooled) # (B, c)
        ### Annotation info
        logits.append(at_out.logits) # these are already argmaxed
        logits.append(annotation_labels.detach())
        ### Annotation losses
        losses.append(self.at_ce_hyper * at_out.loss)

        if self.target:
            tg_out = self.target_encoder(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                labels=target_labels,
                output_full_logits=False
            )
            tg_state = tg_out.last_hidden_state # (B, L, d2)
            tg_state = self.target_state_proj(tg_state) # (B, L, c)
            tg_pooled = tg_out.pooler_output # (B, d2)
            tg_pooled = self.target_pool_proj(tg_pooled) # (B, c)
            target_c_loss = self.contrastive_loss(tg_pooled, at_pooled)
            ### Target info
            logits.append(tg_out.logits)
            logits.append(target_labels.detach())
            logits.append(target_c_loss.detach())
            ### Target losses
            losses.append(self.tg_ce_hyper  * tg_out.loss)
            losses.append(self.tg_cont_hyper * target_c_loss)

        if self.context:
            at_state = self.annotation_state_proj(at_state) # (B, L, d1)
            ct_out = self.context_encoder(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                encoder_hidden_states=at_state,
                encoder_attention_mask=annotation_attention_mask,
                labels=context_labels,
                output_attentions=output_attentions,
                output_full_logits=False
            )
            ct_state = ct_out.last_hidden_state # (B, L, d1)
            ct_state = self.context_state_proj(ct_state) # (B, L, c)
            ct_pooled = ct_out.pooler_output # (B, d1)
            ct_pooled = self.context_pool_proj(ct_pooled) # (B, c)
            ### Context info
            logits.append(ct_out.logits)
            logits.append(context_labels.detach())
            ### Context losses
            losses.append(self.ct_ce_hyper * ct_out.loss)
            if self.target:
                l1_loss = self.l1(ct_state[1:], tg_state[1:]) # remove cls
                logits.append(l1_loss.detach()) # info
                losses.append(self.l1_hyper * l1_loss) # loss

        ### Logits order
        # at_logits, at_labels, tg_logits, tg_labels, tg_contrastive, ct_logits, ct_labels, l1
        ### Loss order
        # at_ce, tg_ce, tg_contrastive, ct_ce, l1
        if target_labels != None or context_labels != None or annotation_labels != None:
            loss = sum(losses)
        else:
            loss = None
        return GeneralModelOutput(
            loss=loss,
            logits=logits,
            attentions=ct_out.attentions if output_attentions else None,
            cross_attentions=ct_out.cross_attentions if output_attentions else None
        )
