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
import copy
import sys


NN = True
numOutputs = 2
randWeight = 0.0

dataType=  sys.argv[1]  		# "4G" "Flocking"  #

explorationSetting = sys.argv[2] 	#	use "state_transitions" for Swarm-LfO or  "obs_transitions" for Dec-Exp-LfO
num_repetitions = 10
inputObsFileName=  dataType +"_demonstrations/"+ dataType +"_observation_transitions" 
inputStateFileName =  dataType + "_demonstrations/" + dataType + "_state_transitions" 


class NeuralNetwork(nn.Module):
	def __init__(self,feature_size, action_size):
		super(NeuralNetwork, self).__init__()
		hidden_size= 50 #250 # 230 # int(2*feature_size)
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(feature_size, hidden_size),
			nn.ReLU(),	
			nn.Linear(int(hidden_size), action_size),

		)
		#self.linear_relu_stack.apply(self.init_weights)
		self.patience=3
		self.min_delta=0.00002
		self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)

	def forward(self, x):
		actions = self.linear_relu_stack(x)
		return actions
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform(m.weight)
			m.bias.data.fill_(0.01)




class EstimatedActionsDataset(Dataset):

	def __init__(self, dataset, transform=None):
		self.annotations = dataset
		shuffle(self.annotations)
		self.transform = transform

	def __len__(self):

		return len(self.annotations)

	def __getitem__(self, idx):

		x= self.annotations[idx][0:-self.num_outputs]
		y= self.annotations[idx][-self.num_outputs:]
		return x, y
		

class SwarmTasksDataset(Dataset):

	def __init__(self, csv_file, root_dir, transform=None):
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
		




def predictMissingActions(fileName, fileName_relative, num_outputs, inverseDynamicsNet):
	dataset = (pd.read_csv(fileName)).to_numpy()
	dataset = dataset.astype(float)
	observations_only = copy.deepcopy(dataset[:,0:-num_outputs])
	pred = np.zeros((len(dataset),num_outputs))
	num_state_vars = int(len(observations_only[0] )/2)

	for i in range(len(dataset)):
		pred[i] = inverseDynamicsNet(torch.from_numpy(dataset[i][0:-num_outputs]).float()).cpu().detach().numpy() 

	dataset_relative = (pd.read_csv(fileName_relative)).to_numpy()
	dataset_relative = dataset_relative.astype(float)
	num_features= int((len(dataset_relative[0])-num_outputs)/2)

	return np.concatenate((dataset_relative[:, 0:num_features], pred), axis =1) , np.concatenate((observations_only[:, 0:num_state_vars], pred), axis =1)




def evaluatePredictedActions(fileName, fileName_relative, num_outputs, inverseDynamicsNet):
	dataset = (pd.read_csv(fileName)).to_numpy()
	dataset = dataset.astype(float)
	observations_only = copy.deepcopy(dataset[:,0:-num_outputs])
	pred = np.zeros((len(dataset),num_outputs))
	MSE = [0 for n in range(num_outputs)]
	avgY = [0 for n in range(num_outputs)]
	num_state_vars = int(len(observations_only[0] )/2)
	error_values = np.zeros((num_outputs, len(dataset)))


	for i in range(len(dataset)):
		pred = inverseDynamicsNet(torch.from_numpy(dataset[i][0:-num_outputs]).float()).cpu().detach().numpy() 
		y = dataset[i][-num_outputs:]

		dist_sq = [0  for n in range(num_outputs)]
		for j in range(num_outputs):
			dist_sq[j]  += (y[j]- pred[j])**2
			MSE[j] += dist_sq[j]  
			error_values[j][i] =  y[j]- pred[j]
			avgY[j]+=y[j]
			
	for j in range(num_outputs):
		#print("RMSE ", j , ": ", math.sqrt(MSE[j]/len(dataset)))
		print("avg action value: ", j, avgY[j]/len(dataset))


	#for i in range(len(dataset)):
	#	print(round(error_values[2][i],4))
	#print("\n")

		







def predictMissingActionsForDecExpLfO(fileName_relative, num_outputs, inverseDynamicsNet):

	dataset_relative = (pd.read_csv(fileName_relative)).to_numpy()
	dataset_relative = dataset_relative.astype(float)
	num_features= int((len(dataset_relative[0])-num_outputs)/2)
	pred = np.zeros((len(dataset_relative),num_outputs))

	for i in range(len(dataset_relative )):
		pred[i] = inverseDynamicsNet(torch.from_numpy(dataset_relative[i][0:-num_outputs]).float()).cpu().detach().numpy() 	

	return np.concatenate((dataset_relative[:, 0:num_features], pred), axis =1)


def predictMissingActionsForLfO(fileName, fileName_relative, num_outputs, inverseDynamicsNet):
	dataset = (pd.read_csv(fileName)).to_numpy()
	dataset = dataset.astype(float)
	observations_only = copy.deepcopy(dataset[:,0:-num_outputs])
	pred = np.zeros((len(dataset),num_outputs))
	num_state_vars = int(len(observations_only[0] )/2)

	for i in range(len(dataset)):
		pred[i] = inverseDynamicsNet(torch.from_numpy(dataset[i][0:-num_outputs]).float()).cpu().detach().numpy() 

	dataset_relative = (pd.read_csv(fileName_relative)).to_numpy()
	dataset_relative = dataset_relative.astype(float)
	num_features= int((len(dataset_relative[0])-num_outputs)/2)

	return np.concatenate((dataset_relative[:, 0:num_features], pred), axis =1) , np.concatenate((observations_only[:, 0:num_state_vars], pred), axis =1)





if True:
	for i in range(num_repetitions):
		if explorationSetting == "obs_transitions":
			inverseDynamicsNetwork= pickle.load(open("IDM-obs\inverseDynamicsNetwork" + str(i) , 'rb'))
			estimated_obs_action= predictMissingActionsForDecExpLfO(inputObsFileName+ str(i) + ".csv" ,numOutputs , inverseDynamicsNetwork)
			with open("IDM-obs_" + dataType+ "_estimated_obs_act_transitions/" + str(i)+ ".csv", "w") as f:
				np.savetxt(f, estimated_obs_action , fmt = '%0.5f' ,delimiter=",")
		else:
			inverseDynamicsNetwork= pickle.load(open("IDM\inverseDynamicsNetwork" + str(i) , 'rb'))
			estimated_obs_action, estimated_state_action = predictMissingActionsForLfO(inputStateFileName + str(i)+".csv",  inputObsFileName+ str(i) + ".csv" ,numOutputs , inverseDynamicsNetwork)

			with open(dataType+ "_estimated_obs_act_transitions/" +  str(i)+ ".csv", "w") as f:
				np.savetxt(f, estimated_obs_action , fmt = '%0.5f' ,delimiter=",")
			with open(dataType+ "_estimated_state_act_transitions/" +  str(i)+ ".csv", "w") as f:
				np.savetxt(f, estimated_state_action , fmt = '%0.5f' ,delimiter=",")

