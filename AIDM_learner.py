from EarlyStopper import EarlyStopper
import numpy as np
import pandas as pd
import os
import time
import random
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import pickle
import math
import sys

test = True
numOutputs = 2 
train = sys.argv[1] # "1" or "0"
explorationSetting = sys.argv[2] # "obs_transitions" or "state_transitions"


if explorationSetting =="state_transitions":
	path = "exploration"
	fileName =  "random_local"

	numFeatures= 10
else: 
	path = "exploration_obs"  
	fileName = "random_local_observation_transitions"  
	numFeatures= 18



num_eps = 10


class NeuralNetwork(nn.Module):
	def __init__(self,feature_size, action_size):
		super(NeuralNetwork, self).__init__()
		hidden_size= 50 
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(feature_size, hidden_size),
			nn.ReLU(),	
			nn.Linear(int(hidden_size), action_size),

		)
		
		self.patience=3
		self.min_delta=0.00
		self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)

	def forward(self, x):
		actions = self.linear_relu_stack(x)
		return actions
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform(m.weight)
			m.bias.data.fill_(0.01)




class SwarmTasksDataset(Dataset):

	def __init__(self, csv_file, root_dir, transform=None):
		print("loading data from", csv_file)
		self.annotations = (pd.read_csv(csv_file)).to_numpy()
		self.annotations = self.annotations.astype(float)
		shuffle(self.annotations)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):

		return len(self.annotations)

	def __getitem__(self, idx):
		x= self.annotations[idx][0:-self.num_outputs]
		y= self.annotations[idx][-self.num_outputs:]
		return x, y
		



def trainSupervised(train_dataloader, test_dataloader, numFeatures  , numOutputs, epochs, lr):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	supervisedNetwork= NeuralNetwork(numFeatures  , numOutputs).to(device)

	loss_fn =MSELoss()
	optimizer = torch.optim.Adam(supervisedNetwork.parameters() , lr= lr) 
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		train_loop(supervisedNetwork, loss_fn, optimizer, train_dataloader)

		if test_dataloader is not None:
			validation_loss = test_loop(test_dataloader, supervisedNetwork, loss_fn)
			if supervisedNetwork.early_stopper.early_stop(validation_loss):             
				break
			#scheduler.step(validation_loss)

	return supervisedNetwork
	


def train_loop(model, loss_fn, optimizer, dataloader):
	size= len(dataloader.dataset)
	avg_loss = 0
	counter = 0
	model.train()
	for batch, (X, y) in enumerate(dataloader):
	
		optimizer.zero_grad()
		# Compute prediction and loss

		pred = model(X.float())
		Loss = MSELoss()
		loss = Loss(pred, y.float())
		# Backpropagation
		loss.backward()
		optimizer.step()

		#if batch % 100 == 0:
		loss, current = loss.item(), batch * len(X)
		avg_loss+= loss
		counter+=1
	avg_loss/=counter
	print(f"avg training loss: {avg_loss:>7f} ")


def test_loop(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss= 0
        model.eval()
	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X.float())
			test_loss += loss_fn(pred, y.float()).item()


	test_loss /= num_batches
	print(f"Test Error Avg loss: {test_loss:>8f} \n")
	return test_loss




def evaluateinverseDynamics(fileName, num_outputs, inverseDynamicsNet):
	dataset = (pd.read_csv(fileName)).to_numpy()
	dataset = dataset.astype(float)
	MSE = [0 for n in range(num_outputs)]
	avgY = [0 for n in range(num_outputs)]
	for i in range(len(dataset)):
		pred = inverseDynamicsNet(torch.from_numpy(dataset[i][0:-num_outputs]).float()).cpu().detach().numpy() 
		y = dataset[i][-num_outputs:]

		dist_sq = [0  for n in range(num_outputs)]
		for j in range(num_outputs):
			dist_sq[j]  += (y[j]- pred[j])**2

			MSE[j] += dist_sq[j]  
			avgY[j]+=y[j]
			
	for j in range(num_outputs):
		print("RMSE ", j , ": ", math.sqrt(MSE[j]/len(dataset)))
		#print("avg action value: ", j, avgY[j]/len(dataset))




if True:
	if train =="1":
		epochs = 300
		lr = 1e-2
		SwarmTasksDataset.num_outputs = numOutputs 
		
		for i in range(num_eps): 
			annotated_data= SwarmTasksDataset(os.path.join(path, fileName+ str(i) +".csv"), 'data') #
			train_len = int(len(annotated_data)*0.8)
			train_set, test_set = torch.utils.data.random_split(annotated_data, [train_len, len(annotated_data) - train_len])
			train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)
			test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)
			inverseDynamicsNetwork= trainSupervised(train_loader, test_loader, numFeatures, numOutputs, epochs, lr)
			if explorationSetting =="state_transitions":
				pickle.dump(inverseDynamicsNetwork , open("IDM/inverseDynamicsNetwork"+str(i) , 'wb'))
			else:
				pickle.dump(inverseDynamicsNetwork , open("IDM-obs/inverseDynamicsNetwork"+str(i) , 'wb'))

	if test:
		if explorationSetting =="state_transitions":
			for i in range(num_eps):
				inverseDynamicsNetwork= pickle.load(open("IDM/inverseDynamicsNetwork"+str(i) , 'rb'))

				evaluateinverseDynamics(os.path.join(path, fileName+ str(num_eps-i-1) +".csv"),numOutputs , inverseDynamicsNetwork)

		else:
			for i in range(num_eps):
				inverseDynamicsNetwork= pickle.load(open("IDM-obs/inverseDynamicsNetwork"+str(i) , 'rb'))
				evaluateinverseDynamics(os.path.join(path, fileName+ str(num_eps-i-1) +".csv"),numOutputs , inverseDynamicsNetwork)		

