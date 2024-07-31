import torch
from torch.nn import CrossEntropyLoss
from transformers import EsmForMaskedLM, EsmTokenizer
from Bio import Align
from Bio.Align import substitution_matrices


# ANSI escape codes for colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def analyze_alignment(label, pred, gap_score=-10):
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = gap_score
    alignments = aligner.align(label, pred)
    best_alignment = alignments[0]
    blosum62 = substitution_matrices.load("BLOSUM62")
    aligned_label, aligned_pred = best_alignment[0], best_alignment[1]
    result = []
    positive_count = 0
    total_aligned = 0
    for a, b in zip(aligned_label, aligned_pred):
        if a == '-' or b == '-':
            result.append((BLUE, a, b, aligner.open_gap_score))
        else:
            score = blosum62[a, b]
            total_aligned += 1
            if score > 0:
                result.append((GREEN, a, b, score))
                positive_count += 1
            elif score < 0:
                result.append((RED, a, b, score))
            else:
                result.append((YELLOW, a, b, score))
                positive_count += 1
    percent_positive = (positive_count / total_aligned) * 100 if total_aligned > 0 else 0
    return result, best_alignment.score, percent_positive

def print_colored_alignment(result):
    label_line = ""
    pred_line = ""
    score_line = ""

    for color, a, b, score in result:
        label_line += f"{color}{a}{RESET}"
        pred_line += f"{color}{b}{RESET}"
        if a == '-' or b == '-':
            score_line += f"{color}-{RESET}"
        elif score > 0:
            score_line += f"{color}+{RESET}"
        elif score < 0:
            score_line += f"{color}-{RESET}"
        else:
            score_line += f"{color}0{RESET}"

    print("Label:", label_line)
    print("Pred :", pred_line)
    print("Score:     ", score_line)


def mask_tokens(inputs, labels, mask_probability, tokenizer, sectional_masking=False):
    probability_matrix = torch.full(inputs.shape, mask_probability)
    special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
    special_token_ids = [
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id
    ]
    for token_id in special_token_ids:
        special_tokens_mask |= (inputs == token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    if sectional_masking:
        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        for i in range(inputs.shape[0]):
            non_special_indices = (~special_tokens_mask[i]).nonzero(as_tuple=True)[0]
            if len(non_special_indices) > 0:
                # Calculate expected number of masked tokens
                expected_masked = int(len(non_special_indices) * mask_probability)
                if expected_masked == 0:
                    expected_masked = 1  # Ensure at least one token is masked
                # Randomly choose start index
                start_idx = non_special_indices[torch.randint(0, len(non_special_indices) - expected_masked + 1, (1,))]
                # Set end index based on expected number of masked tokens
                end_idx = min(start_idx + expected_masked, len(non_special_indices))
                masked_indices[i, start_idx:end_idx] = True
    else:
        # Original random masking
        masked_indices = torch.bernoulli(probability_matrix).bool()
    # Ensure at least one token is masked
    if not masked_indices.any():
        non_special_indices = (~special_tokens_mask).nonzero(as_tuple=True)[1]
        if len(non_special_indices) > 0:
            random_index = non_special_indices[torch.randint(0, len(non_special_indices), (1,))]
            masked_indices[0, random_index] = True
    labels[~masked_indices] = -100  # Set labels for non-masked tokens to -100
    inputs[masked_indices] = tokenizer.mask_token_id  # Replace masked tokens with mask token ID
    return inputs, labels


def sequence_acc(pred, label):
    return sum(p == l for p, l in zip(pred, label)) / len(label)


class EsmGenerator(EsmForMaskedLM):
    def __init__(self, config, add_pooler_layer=False):
        super().__init__(config)
        self.config = config
        self.ce = CrossEntropyLoss()
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8m_UR50D')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.special_token_ids = [self.pad_token_id, self.cls_id, self.eos_id, self.mask_token_id]
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def decode_seq(self, ids):
        return self.tokenizer.decode(ids).replace(' ', '').replace('<mask>', '-').replace('<cls>', '').replace('<eos>', '')

    def sort_and_update(self, sorted_probs, n, k, ids, samp_method):
        # sorted_probs, dictionary - int: (float, int) - index: (probability tensors (1, v), max probability)
        # n, int - number of tokens to keep
        # k, int - choose within the topk probs per token
        # ids, tensor - replace with predicted tokens
        # Sort the dictionary by the probabilities
        sorted_items = sorted(sorted_probs.items(), key=lambda x: x[1][1], reverse=True)
        top_n_items = sorted_items[:min(n, len(sorted_items))]
        for idx, (odds, top_token) in top_n_items:
            if "topk" in samp_method:
                token = self.sample_top_k(odds, k)
                ids[idx] = token
            if "nuc" in samp_method:
                token = self.sample_nucleus(odds, k)
                ids[idx] = token
        return ids

    def sample_nucleus(self, probs, p):
        if probs.numel() == 0:
            raise ValueError("Input tensor is empty")
        
        mask = probs >= p
        masked_tensor = probs * mask

        if torch.all(masked_tensor == 0):
            chosen_idx = torch.argmax(probs).item()
        else:
            chosen_idx = torch.multinomial(masked_tensor, num_samples=1).item()

        return chosen_idx

    def sample_top_k(self, logits, k):
        top_k_probs, top_k_indices = torch.topk(logits, k)
        chosen_idx = torch.multinomial(top_k_probs, num_samples=1)
        return top_k_indices[chosen_idx.item()].item()

    def generate(
            self,
            template_ids,
            template_mask,
            n=1,
            k=1,
            p=0.1,
            entropy=True,
            temperature=1.0,
            device='cpu',
            view=False,
            samp_method=None
    ):
        # sequence, string - input sequence
        # mask_probability, float - probability of masking a token
        # n, int - number of predictions to keep each forward pass
        # k, int - choose within the topk probs per token
        # entropy, bool - whether to use entropy-based sampling (softmax) or just logits
        # temperature, float - softmax temperature
        # device, string - 'cpu' or 'cuda'
        # view, bool - whether to print progress

        assert (p is None) != (k is None), "Exactly one of 'p' or 'k' must be specified, not both or neither."
        assert samp_method in ["topk", "nuc"], "samp_method must be either 'topk' or 'nuc'."
        assert (samp_method == "topk" and k is not None) or (samp_method == "nuc" and p is not None), "samp_method must match the specified parameter (k for topk, p for nuc)."

        template_ids, template_mask = template_ids.to(device), template_mask.to(device)
        seq_len = len(template_ids[0])
        if view:
            print(self.decode_seq(template_ids[0]))

        if n > seq_len:
            seq_output = self.esm(
                input_ids=template_ids,
                attention_mask=template_mask
            ).last_hidden_state
            logits = self.lm_head(seq_output)
            current_seq = self.decode_seq(logits.argmax(dim=-1)[0])
            if view:
                print(current_seq)
        else:
            for _ in range(seq_len):
                current_seq = self.decode_seq(template_ids[0])
                if view:
                    print(f"\r{current_seq}", end="", flush=True)
                mask_indices = (template_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1] # Find masked positions
                if len(mask_indices) == 0:
                    break  # No more masks to fill

                seq_output = self.esm(
                    input_ids=template_ids,
                    attention_mask=template_mask
                ).last_hidden_state
                logits = self.lm_head(seq_output) # get logits

                if entropy:
                    probs = (logits / temperature).softmax(dim=-1).squeeze(0)
                else:
                    probs = logits.squeeze(0)
            
                sorted_probs = {}
                for idx in mask_indices:
                    mask_fill_odds = probs[idx]
                    sorted_probs[idx.item()] = (mask_fill_odds, mask_fill_odds.max().item())
    
                template_ids = self.sort_and_update(sorted_probs, n, k if samp_method == "topk" else p, template_ids.squeeze(0), samp_method)
                template_ids = template_ids.unsqueeze(0)
        return current_seq
    

if __name__ == '__main__':
    model_path = 'facebook/esm2_t33_650M_UR50D'
    sequence = 'MKNNLSKNKKVPAIRDKGHKAGKGFKTLSKGLGLPVSSVGSITRKWKAYRTTVNLPRPGQPFKISSRAKGQGVAVAYSVSLT'
    mask_prob = 0.90
    n = 1
    k = 20
    entropy = True
    temperature = 0.7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    view = True

    model = EsmGenerator.from_pretrained(model_path).eval().to(device)
    tokenizer = EsmTokenizer.from_pretrained(model_path)

    template = tokenizer(sequence, add_special_tokens=True, return_tensors='pt')
    template_ids, template_mask = template['input_ids'], template['attention_mask']
    labels = template_ids.clone()
    template_ids, labels = mask_tokens(template_ids, labels, mask_prob, tokenizer, sectional_masking=True)

    with torch.no_grad():
        generated_seq = model.generate(
                template_ids.clone(),
                template_mask,
                n=n,
                k=k,
                temperature=temperature,
                device=device,
                entropy=True,
                view=view
        )
    try:
        result, score, percent_positive = analyze_alignment(sequence, generated_seq, gap_score=-100)
        acc = sequence_acc(result, labels)
        print("\n\nColored alignment analysis:")
        print_colored_alignment(result)
        print(f"\nTotal alignment score: {score}")
        print(f"Normalized score: {score / len(sequence):.3f}")
        print(f"Sequence accuracy: {acc*100:.3f}")
        print(f"Percent positive: {percent_positive:.2f}%")
        print("\nColor legend:")
        print(f"{GREEN}Green: Positive scoring substitution{RESET}")
        print(f"{RED}Red: Negative scoring substitution{RESET}")
        print(f"{YELLOW}Yellow: Zero-scoring match or substitution{RESET}")
        print(f"{BLUE}Blue: Gap{RESET}")
    except:
        print('Special token in output')
