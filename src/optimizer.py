# Credits : https://github.com/mgrankin/over9000

# Lookahead implementation from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lookahead.py



""" Lookahead Optimizer Wrapper.

Implementation modified from: https://github.com/alphadl/lookahead.pytorch

Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610

"""

import torch

from torch.optim.optimizer import Optimizer

from collections import defaultdict


class Lookahead(Optimizer):

    def __init__(self, base_optimizer, alpha=0.5, k=6):

        if not 0.0 <= alpha <= 1.0:

            raise ValueError(f"Invalid slow update rate: {alpha}")

        if not 1 <= k:

            raise ValueError(f"Invalid lookahead steps: {k}")

        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)

        self.base_optimizer = base_optimizer

        self.param_groups = self.base_optimizer.param_groups

        self.defaults = base_optimizer.defaults

        self.defaults.update(defaults)

        self.state = defaultdict(dict)

        # manually add our defaults to the param groups

        for name, default in defaults.items():

            for group in self.param_groups:

                group.setdefault(name, default)



    def update_slow(self, group):

        for fast_p in group["params"]:

            if fast_p.grad is None:

                continue

            param_state = self.state[fast_p]

            if "slow_buffer" not in param_state:

                param_state["slow_buffer"] = torch.empty_like(fast_p.data)

                param_state["slow_buffer"].copy_(fast_p.data)

            slow = param_state["slow_buffer"]

            slow.add_(fast_p.data - slow, alpha=group["lookahead_alpha"])

            fast_p.data.copy_(slow)



    def sync_lookahead(self):

        for group in self.param_groups:

            self.update_slow(group)



    def step(self, closure=None):

        # print(self.k)

        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)

        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:

            group["lookahead_step"] += 1

            if group["lookahead_step"] % group["lookahead_k"] == 0:

                self.update_slow(group)

        return loss



    def state_dict(self):

        fast_state_dict = self.base_optimizer.state_dict()

        slow_state = {(id(k) if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()}

        fast_state = fast_state_dict["state"]

        param_groups = fast_state_dict["param_groups"]

        return {

            "state": fast_state,

            "slow_state": slow_state,

            "param_groups": param_groups,

        }



    def load_state_dict(self, state_dict):

        fast_state_dict = {

            "state": state_dict["state"],

            "param_groups": state_dict["param_groups"],

        }

        self.base_optimizer.load_state_dict(fast_state_dict)



        # We want to restore the slow state, but share param_groups reference

        # with base_optimizer. This is a bit redundant but least code

        slow_state_new = False

        if "slow_state" not in state_dict:

            print("Loading state_dict from optimizer without Lookahead applied.")

            state_dict["slow_state"] = defaultdict(dict)

            slow_state_new = True

        slow_state_dict = {

            "state": state_dict["slow_state"],

            "param_groups": state_dict["param_groups"],  # this is pointless but saves code

        }

        super(Lookahead, self).load_state_dict(slow_state_dict)

        self.param_groups = self.base_optimizer.param_groups  # make both ref same container

        if slow_state_new:

            # reapply defaults to catch missing lookahead specific ones

            for name, default in self.defaults.items():

                for group in self.param_groups:

                    group.setdefault(name, default)
 
 
 # Credits : https://github.com/mgrankin/over9000
import torch, math
from torch.optim.optimizer import Optimizer

# RAdam + LARS + GC
class Ralamb(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, gc_conv_only=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("Ralamb does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True), alpha=-1)

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # trying yogi
                # grad_squared = grad.mul(grad)
                # exp_avg_sq.mul_(beta2).addcmul_(-(1 - beta2), torch.sign(exp_avg_sq - grad_squared), grad_squared)

                state["step"] += 1
                buffered = self.buffer[int(state["step"] % 10)]

                if state["step"] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = radam_step_size

                # if group["weight_decay"] != 0:
                #    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    radam_step.addcdiv_(exp_avg, denom, value=-radam_step_size * group["lr"])
                    # GC
                    G_grad = exp_avg.div(denom)
                else:
                    radam_step.add_(exp_avg, alpha=-radam_step_size * group["lr"])
                    # GC
                    G_grad = exp_avg

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state["weight_norm"] = weight_norm
                state["adam_norm"] = radam_norm
                state["trust_ratio"] = trust_ratio

                # GC operation for Conv layers and FC layers
                if G_grad.dim() > self.gc_gradient_threshold:
                    G_grad.add_(G_grad.mean(dim=tuple(range(1, G_grad.dim())), keepdim=True), alpha=-1)

                p_data_fp32.add_(G_grad, alpha=-radam_step_size * group["lr"] * trust_ratio)

                p.data.copy_(p_data_fp32)

        return loss
 
 
 # Credits: https://huggingface.co/transformers/_modules/transformers/optimization.html
 from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)