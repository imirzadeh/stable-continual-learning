import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_sgd.models import MLP, ResNet18
from stable_sgd.data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks, get_split_cifar100_tasks
from stable_sgd.utils import parse_arguments, DEVICE, init_experiment, end_experiment, log_metrics, log_hessian, save_checkpoint


def train_single_epoch(net, optimizer, loader, criterion, task_id=None):
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	net.train()
	
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		if task_id:
			pred = net(data, task_id)
		else:
			pred = net(data)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
	return net


def eval_single_epoch(net, loader, criterion, task_id=None):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			# for cifar head
			if task_id:
				output = net(data, task_id)
			else:
				output = net(data)
			test_loss += criterion(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss}


def get_benchmark_data_loader(args):
	"""
	Returns the benchmark loader which could be either of these:
	get_split_cifar100_tasks, get_permuted_mnist_tasks, or get_rotated_mnist_tasks
	
	:param args:
	:return: a function which when called, returns all tasks
	"""
	if args.dataset == 'perm-mnist' or args.dataset == 'permuted-mnist':
		return get_permuted_mnist_tasks
	elif args.dataset == 'rot-mnist' or args.dataset == 'rotation-mnist':
		return get_rotated_mnist_tasks
	elif args.dataset == 'cifar-100' or args.dataset == 'cifar100':
		return get_split_cifar100_tasks
	else:
		raise Exception("Unknown dataset.\n"+
						"The code supports 'perm-mnist, rot-mnist, and cifar-100.")


def get_benchmark_model(args):
	"""
	Return the corresponding PyTorch model for experiment
	:param args:
	:return:
	"""
	if 'mnist' in args.dataset:
		if args.tasks == 20 and args.hiddens < 256:
			print("Warning! the main paper MLP with 256 neurons for experiment with 20 tasks")
		return MLP(args.hiddens, {'dropout': args.dropout}).to(DEVICE)
	elif 'cifar' in args.dataset:
		return ResNet18(config={'dropout': args.dropout}).to(DEVICE)
	else:
		raise Exception("Unknown dataset.\n"+
						"The code supports 'perm-mnist, rot-mnist, and cifar-100.")


def run(args):
	"""
	Run a single run of experiment.
	
	:param args: please see `utils.py` for arguments and options
	"""
	# init experiment
	acc_db, loss_db, hessian_eig_db = init_experiment(args)

	# load benchmarks and model
	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
	print("loaded all tasks!")
	model = get_benchmark_model(args)

	# criterion
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	
	for current_task_id in range(1, args.tasks+1):
		print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
		train_loader = tasks[current_task_id]['train']
		lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
		
		for epoch in range(1, args.epochs_per_task+1):
			# 1. train and save
			optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
			train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
			time += 1
			print("Finished training task {}".format(current_task_id))
			print("Going to collect continual learning metrics (e.g., accuracy, loss, ....)")

			# 2. evaluate on all tasks up to now, including the current task
			for prev_task_id in range(1, current_task_id+1):
				# 2.0. only evaluate once a task is finished
				if epoch == args.epochs_per_task:
					model = model.to(DEVICE)
					val_loader = tasks[prev_task_id]['test']
					
					# 2.1. compute accuracy and loss
					metrics = eval_single_epoch(model, val_loader, criterion, prev_task_id)
					acc_db, loss_db = log_metrics(metrics, time, prev_task_id, acc_db, loss_db)
					
					# 2.2. (optional) compute eigenvalues and eigenvectors of Loss Hessian
					if prev_task_id == current_task_id and args.compute_eigenspectrum:
						hessian_eig_db = log_hessian(model, val_loader, time, prev_task_id, hessian_eig_db)
						
					# 2.3. save model parameters
					save_checkpoint(model, time)

	end_experiment(args, acc_db, loss_db, hessian_eig_db)


if __name__ == "__main__":
	args = parse_arguments()
	run(args)