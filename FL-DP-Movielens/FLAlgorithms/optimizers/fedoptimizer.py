from torch.optim import Optimizer
import torch
import math

class peravgAdamOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(peravgAdamOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(peravgAdamOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None, beta=0):
        loss = None
        if closure is not None:
            loss = closure
        # weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # print('grad: {}\t'.format(grad))
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]  # 之前的step累计数据

                # state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)  # [batch, seq]
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']  # 上次的r与s
                if amsgrad:
                    # asmgrad优化方法是针对Adam的改进，通过添加额外的约束，使学习率始终为正值。
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # 序号对应最后一幅图中序号
                if group['weight_decay'] != 0:  # 进行权重衰减(实际是L2正则化）
                    # 6. grad(t)=grad(t-1)+ weight*p(t-1)
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                # 7.计算m(t): m(t)=beta_1*m(t-1)+(1-beta_1)*grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 8.计算v(t): v(t)= beta_2*v(t-1)+(1-beta_2)*grad^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    # 迭代改变max_exp_avg_sq的值（取最大值），传到下一次，保留之前的梯度信息。
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    # 计算sqrt(v(t))+epsilon
                    # sqrt(v(t))+eps = denom = sqrt(v(t))/sqrt(1-beta_2^t)+eps
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # step_size=lr/bias_correction1=lr/(1-beta_1^t)

                if (beta != 0):
                    # p.data.add_(-beta, d_p)
                    step_size = beta / bias_correction1
                else:
                    # p.data.add_(-group['lr'], d_p)
                    step_size = group['lr'] / bias_correction1

                # step_size = group['lr'] / bias_correction1

                # p(t)=p(t-1)-step_size*m(t)/denom
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        return group['params']