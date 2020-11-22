"""Yogi: Extension of yogi adaptive nonconvex optimizer in Keras.

Implementation of Additive Averaging.
m_t+1 = beta1*m_t + (1-beta1)*g_t
v_t+1 = v_t + sign(g_t-v_t)(g_t^2)

Experiments show better performance across NLP and Vision tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import math
import torch
from torch.optim import Optimizer


class Yogi(Optimizer):
	"""Implements Yogi algorithm.

	See Algorithm 2 of go/yogi-opt.
	"""

	def __init__(self,
				 params,
				 lr=1e-2,
				 betas=(0.9, 0.999),
				 eps=1e-3,
				 regularization=(0, 0),
				 initial_accumulator_value=1e-6):
		"""Construct a new Yogi optimizer.

		Args:
		  params: iterable of parameters to optimize or dicts defining parameter
			groups (iterable)
		  lr: learning rate (float, optional) (default: 1e-2)
		  betas: coefficients used for computing running averages of gradient and
			its square (Tuple[float, float], optional) (default: (0.9, 0.999))
		  eps: A constant trading off adaptivity and noise.
			(float, optional) (default: 1e-3)
		  regularization: L1 and L2 penalty. Must be greater than or equal to zero.
			(Tuple[float, float], optional) (default: (0, 0))
		  initial_accumulator_value: The starting value for accumulators.
			Only positive values are allowed. (float, optional) (default: 1.0)
		"""
		if not 0.0 <= lr:
			raise ValueError('Invalid learning rate: {}'.format(lr))
		if not 0.0 <= eps:
			raise ValueError('Invalid epsilon value: {}'.format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
		if not 0.0 <= regularization[0]:
			raise ValueError('Invalid L1 penalty: {}'.format(regularization[0]))
		if not 0.0 <= regularization[1]:
			raise ValueError('Invalid L2 penalty: {}'.format(regularization[1]))
		defaults = dict(lr=lr, betas=betas, eps=eps, regularization=regularization)
		super(Yogi, self).__init__(params, defaults)
		self.v_init = initial_accumulator_value
		# self.num_dims = sum([sum([p.data.numel() for p in group['params']])
		# for group in self.param_groups])

	def step(self, closure=None, l1_penalty=None, l2_penalty=None):
		"""Performs a single optimization step.

		Args:
		  closure: A closure that reevaluates the model and returns the loss.
			(callable, optional)

		Returns:
		  loss: value of loss

		Raises:
		  RuntimeError: if applied on sparse matrices
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					raise RuntimeError(
					  'Yogi does not support sparse gradients, please consider SparseAdam instead'
					)

				state = self.state[p]

				# State initialization
				if not state:
					state['step'] = 0
					# Exponential moving average of gradient values
					state['exp_avg'] = torch.zeros_like(p.data)
					# Exponential moving average of squared gradient values
					state['exp_avg_sq'] = torch.zeros_like(p.data) + self.v_init

				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				beta1, beta2 = group['betas']
				if not l1_penalty:
					l1_penalty = group['regularization'][0]
				if not l2_penalty:
					l2_penalty = group['regularization'][1]

				state['step'] += 1

				# if state['step'] < 1000:
				#	self.v_init += (grad*grad).sum().item()
				# return loss
				#
				# if state['step'] == 1000:
				#	state['exp_avg_sq'] = torch.zeros_like(p.data) +
				# .                               self.v_init/(1000*self.num_dims)
				#	print(self.v_init, self.num_dims)

				# Decay the first and second moment running average coefficient
				# pdb.set_trace()
				exp_avg.mul_(beta1).add_(1 - beta1, grad)

				grad2 = grad.mul_(grad)
				sign = (exp_avg_sq - grad2).sign_()
				exp_avg_sq.addcmul_(beta2 - 1, sign, grad2)
				del grad2, sign, grad

				denom = exp_avg_sq.sqrt().add_(group['eps'])

				bias_correction1 = 1 - beta1**state['step']
				bias_correction2 = 1 - beta2**state['step']
				step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

				# Variable update
				# Step 1: Gradient descent
				p.data.addcdiv_(-step_size, exp_avg, denom)
				# Step 2: Prox operator
				if l1_penalty > 0:
					per_coord_lr = step_size / denom
					l1_per_coord_lr = l1_penalty * per_coord_lr  # step_size # per_coord_lr
					# p.data = (p.data.abs() > l1_per_coord_lr).type(p.data.dtype) * (p.data - p.data.sign()*l1_per_coord_lr)
					p.data = (p.data > l1_per_coord_lr).type(p.data.dtype) * (p.data - l1_per_coord_lr)
					if l2_penalty > 0:
						p.data.div_(1 + l2_penalty * per_coord_lr)
				elif l2_penalty > 0:
					per_coord_lr = step_size / denom
					p.data.div_(1 + l2_penalty * per_coord_lr)

		return loss
