import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from https://github.com/SimiaoZuo/MoEBERT/blob/master/src/transformers/moebert/moe_layer.py
class LoadBalancingLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wBAL = torch.tensor(config.wBAL, requires_grad=False)
        self.num_experts = config.num_experts

    def forward(self, router_logits: torch.Tensor) -> torch.Tensor: 
        # enforces experts should not be used widely more than another
        num_experts = self.num_experts
        wBAL = torch.abs(self.wBAL)
        if isinstance(router_logits, tuple):
            router_logits = torch.cat(router_logits, dim=0) # batch_size * num_hidden_layers, num_experts
        # can also be batchsize * num_tasks * num_hidden_layers, num_experts
        router_logits = router_logits.reshape(-1, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        gate = torch.argmax(router_probs, dim=-1)
        num_tokens = F.one_hot(gate, num_experts).gt(0).sum(0)
        p = router_probs.mean(0)
        temp = num_tokens.float()
        f = temp / temp.sum(0, keepdim=True) 
        return wBAL * num_experts * torch.sum(p * f)
    

# https://github.com/UMass-Foundation-Model/Mod-Squad/blob/1d17e81d090ac7e1a66dd420194c0b7679d820a4/util/AutomaticWeightedLoss.py
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, params=None):
        super(AutomaticWeightedLoss, self).__init__()
        if params == None:
            self.params = torch.ones(num, requires_grad=False)
        else:
            params = [float(param) for param in params]
            self.params = torch.tensor(params, requires_grad=False)

    def forward(self, losses, router_labels):
        loss_sum = 0
        for i, loss in enumerate(losses):
            param = self.params[router_labels[i]] ** 2
            loss_sum += 0.5 / param * loss + torch.log(1 + param)
        return loss_sum


# Adapted from https://github.com/UMass-Foundation-Model/Mod-Squad/blob/1d17e81d090ac7e1a66dd420194c0b7679d820a4/parallel_linear/parallel_experts/moe.py#L25
class MILoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wMI = torch.tensor(config.wMI, requires_grad=False)
        self.MI_task_gate = torch.zeros(config.num_tasks, config.num_experts)
        self.num_experts = config.num_experts
        self.topk = config.topk
        self.token_moe = config.token_moe

    def update_MI_task_gate(self, probs, router_labels):
        """
        Update the MI_task_gate matrix with probabilities of selecting each expert for the current task.

        Args:
        probs (torch.Tensor): The probabilities of selecting each expert for the current task.
                            It should have shape [batch_size * seq_len, num_experts].
        router_labels (torch.Tensor): The labels indicating the task for each example in the batch.
                                    It should have shape [batch_size * seq_len].

        Returns:
        None
        """
        for task in router_labels:
            self.MI_task_gate[task] += probs[router_labels == task].sum(0)

    def calculate_mutual_information_loss(self):
        """
        Calculate the mutual information loss.

        Returns:
        torch.Tensor: The calculated mutual information loss.
        """
        MI_gate = self.MI_task_gate.clone()
        tot = MI_gate.sum() / self.topk
        MI_gate = MI_gate / (tot + 0.0001)
        P_TI = torch.sum(MI_gate, dim=1, keepdim=True) + 0.0001
        P_EI = torch.sum(MI_gate, dim=0, keepdim=True) + 0.0001

        MI_loss = -(MI_gate * torch.log(MI_gate / P_TI / P_EI + 0.0001)).sum()
        return self.wMI * MI_loss

    def call_update(self, router_logits, router_labels):
        router_logits = router_logits.detach().cpu()
        router_labels = router_labels.detach().cpu()
        probs = router_logits.softmax(dim=-1)

        if self.token_moe:
            # router_logits (batch_size, seq_len, num_experts)
            # router_labels (batch_size, )
            # reshape to 
            # router_logits (batch_size * seq_len, num_experts)
            # expand to
            # router_labels (batch_size * seq_len, )
            router_labels = router_labels.unsqueeze(1).expand(-1, probs.size(1)).contiguous().view(-1)
        probs = probs.view(-1, self.num_experts)
        
        self.update_MI_task_gate(probs, router_labels)

    def forward(self, router_logits: torch.Tensor, router_labels: torch.Tensor) -> torch.Tensor:
        if isinstance(router_logits, tuple):
            for layer_router_logits in router_logits:
                self.call_update(layer_router_logits, router_labels)
        else:
            self.call_update(router_logits, router_labels)
        return self.calculate_mutual_information_loss()


def specified_expert_loss(router_logits: torch.Tensor, router_labels: torch.Tensor) -> float: # TODO update
    # enforces on average the router should route examples to the correct specified expert given the known origin of the input
    if router_logits is None:
        return 0
    if isinstance(router_logits, tuple):
        batch_size, num_experts = router_logits[0].shape
        router_logits = torch.stack(router_logits, dim=2).transpose(1, 2) # batch_size, num_hidden_layers, num_experts
    else:
        print('Must be tuple of all layers router logits')
    
    avg_logits = router_logits.mean(dim=1)
    return F.cross_entropy(avg_logits, router_labels)


### tests
if __name__ == 'main':
    pass ### TODO
