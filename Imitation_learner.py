from Agent import Agent
from EarlyStopper import EarlyStopper
import numpy as np
import pandas as pd
import os
import time
import random
import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import pickle
import copy
import math
import sys




dataType=  sys.argv[1]  		# "4G_light" "Flocking"  #
explorationSetting = sys.argv[2] 	#  use "state_transitions" for Swarm-LfO or  "obs_transitions" for Dec-Exp-LfO
LfO_setting = sys.argv[3] 		# "LfO" or "LfD" 
train = sys.argv[4]



if LfO_setting == "LfD" :
	explorationSetting  = ""

dimension= 20
numberOfAgents = 20
numOutputs = 2
num_repetitions = 10
num_eval_runs= 10
path_prefix=""



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


class SwarmTasksDataset(Dataset):

	def __init__(self, csv_file, root_dir, transform=None):
		print("loading data from ", csv_file)
		self.annotations = (pd.read_csv(csv_file)).to_numpy()
		self.annotations = self.annotations.astype(float)
		shuffle(self.annotations)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):

		return len(self.annotations)

	def __getitem__(self, idx):
		x= self.annotations[idx][0:self.num_inputs]
		y= self.annotations[idx][-self.num_outputs:]
		return x, y
		



def trainSupervised(train_dataloader, test_dataloader, numFeatures  , numOutputs, epochs, lr):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	supervisedNetwork= NeuralNetwork(numFeatures  , numOutputs).to(device)

	loss_fn = MSELoss() #L1Loss() #MSELoss()
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

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



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




def evaluateLearntFlocking(dimension, numberOfAgents, imitationNetworkFileName, num_repetitions, num_eval_runs, time_steps, LfO_setting ):


	Agent.dimension= dimension
	print("Distance travelled \t Avg distance to neighbours \t Avg # neighbours \t Avg graph components")

	for i in range(num_repetitions):
		imitationNetwork = pickle.load(open(imitationNetworkFileName+ str(i) , 'rb'))

		agentList = Agent(0,0, dimension , dimension ,numberOfAgents)
	
		target_points =np.array([[dimension, dimension]])
		Agent.target_points = target_points
		Agent.light_intenisity =  320000 # 320000 lux @ 0.05m  == 800 lux @ 1m 

		for e in range(num_eval_runs):
			Agent.target_points[0][0] =np.random.uniform(0, dimension)  
			Agent.target_points[0][1] = dimension
			Agent.x_landmark = Agent.target_points[:,0]
			Agent.y_landmark = Agent.target_points[:,1]

			#initialise swarm member within an square of length d/6 & with random center
			offset_x = 0 #np.random.uniform(0, 9*dimension/10)
			offset_y = 0 #np.random.uniform(0, dimension/2)	
			agentList.reset(offset_x , offset_y, offset_x + 1.5, offset_y + 1.5)
			#init_center = [np.mean(agentList.x) , np.mean(agentList.y)]
			init_dist = math.sqrt((np.mean(agentList.x) - Agent.target_points[0][0])**2 + (np.mean(agentList.y)- Agent.target_points[0][1])**2)
			total_neighbour_distance = 0
			total_num_negibours = 0
			total_num_graph_components = 0

			NewLocationX= np.zeros(numberOfAgents)
			NewLocationY= np.zeros(numberOfAgents)
			NewVelX = np.zeros(numberOfAgents)
			NewVelY = np.zeros(numberOfAgents)	

			for t in range(time_Steps):

				agentList.x_last = copy.deepcopy(agentList.x)
				agentList.y_last = copy.deepcopy(agentList.y)	
	
				for b in range(numberOfAgents):

					observation = np.array(agentList.get_relative_obs(b)) 
					action = imitationNetwork(torch.from_numpy(observation).float()).cpu().detach().numpy()
					MNewLocation = [agentList.x[b] + action[0] , agentList.y[b] + action[1] ]

					# an agent body can't cross the wall
					MNewLocation = agentList.FixBoundaryConditionStickToBorder(MNewLocation[0] ,MNewLocation[1],b)  #agentList.FixBoundaryCondition(MNewLocation[0] ,MNewLocation[1])
					NewLocationX[b] = MNewLocation[0]
					NewLocationY[b] = MNewLocation[1]
					NewVelX[b] = action[0]
					NewVelY[b] = action[1]

				for b in range(numberOfAgents):
					agentList.x[b] = NewLocationX[b] 
					agentList.y[b] = NewLocationY[b]
					agentList.orientation[b] = math.atan2(NewVelY[b],NewVelX[b]) # between -pi, pi
					agentList.avoidCollisions(b) 		# overlapping between two agents' bodies can't happen in 2D environment
					agentList.velocityX[b] = NewVelX[b]
					agentList.velocityY[b] = NewVelY[b]


				# evaluation metrics
				for b in range(numberOfAgents):
					[neighbour_distance_x,neighbour_distance_y, num_negibours] = agentList.getAverageNeighbourDistance(b, agentList.cohesionRange)
					total_neighbour_distance += math.sqrt(neighbour_distance_x**2+neighbour_distance_x**2)
					total_num_negibours += num_negibours 

				total_num_graph_components += len(agentList.numberOfGraphComponents())

					
				agentList.step += 1
				#agentList.render()


			#final_center = [np.mean(agentList.x) , np.mean(agentList.y)]
			final_dist = math.sqrt((np.mean(agentList.x) - Agent.target_points[0][0])**2 + (np.mean(agentList.y)- Agent.target_points[0][1])**2)
			#distance_travelled = math.sqrt((final_center[0] - init_center[0])**2  + (final_center[1] - init_center[1])**2)
			distance_change = final_dist - init_dist 
			print(distance_change , "\t" , total_neighbour_distance/(time_Steps*numberOfAgents) , "\t", total_num_negibours/(time_Steps*numberOfAgents), "\t" , total_num_graph_components/time_Steps )



def evaluateLearnt4G(dimension, numberOfAgents, imitationNetworkFileName, num_repetitions, num_eval_runs, time_steps, LfO_setting ):

	Agent.dimension= dimension

	for i in range(num_repetitions):
		print(imitationNetworkFileName+ str(i))
		imitationNetwork = pickle.load(open(imitationNetworkFileName+ str(i) , 'rb'))

		agentList = Agent(0,0, dimension , dimension ,numberOfAgents)
	
		target_points = np.array([[dimension/4,dimension/4] , [3*dimension/4,dimension/4], [3*dimension/4,3*dimension/4], [dimension/4,3*dimension/4]])
		Agent.target_points = target_points


		for e in range(num_eval_runs):
			Agent.x_landmark = Agent.target_points[:,0]
			Agent.y_landmark = Agent.target_points[:,1]

			#initialise swarm members
			agentList.reset()
			NewLocationX= np.zeros(numberOfAgents)
			NewLocationY= np.zeros(numberOfAgents)
			NewVelX = np.zeros(numberOfAgents)
			NewVelY = np.zeros(numberOfAgents)	

			for t in range(time_Steps):

				agentList.x_last = copy.deepcopy(agentList.x)
				agentList.y_last = copy.deepcopy(agentList.y)	
	
				for b in range(numberOfAgents):

					observation = np.array(agentList.get_relative_obs(b)) 
					action = imitationNetwork(torch.from_numpy(observation).float()).cpu().detach().numpy()
					MNewLocation = [agentList.x[b] + action[0] , agentList.y[b] + action[1] ]

					# an agent body can't cross the wall
					MNewLocation = agentList.FixBoundaryConditionStickToBorder(MNewLocation[0] ,MNewLocation[1],b)  
					NewLocationX[b] = MNewLocation[0]
					NewLocationY[b] = MNewLocation[1]
					NewVelX[b] = action[0]
					NewVelY[b] = action[1]

				for b in range(numberOfAgents):
					agentList.x[b] = NewLocationX[b] 
					agentList.y[b] = NewLocationY[b]
					agentList.orientation[b] = math.atan2(NewVelY[b],NewVelX[b]) # between -pi, pi
					agentList.avoidCollisions(b) 		# overlapping between two agents' bodies can't happen in 2D environment
					agentList.velocityX[b] = NewVelX[b]
					agentList.velocityY[b] = NewVelY[b]

				
				agentList.step += 1
				#agentList.render()

			# evaluation metrics
			agents_per_landmark = np.zeros(4)
			for t in range(len(target_points)):			
				for b in range(len(agentList.x)):
					if math.sqrt((agentList.x[b]-target_points[t][0])**2 + (agentList.y[b]-target_points[t][1])**2) < Agent.bodyRadius*10:
						agents_per_landmark[t]+=1 
			print(agents_per_landmark[0], "\t", agents_per_landmark[1], "\t", agents_per_landmark[2], "\t", agents_per_landmark[3], "\t")



		


if explorationSetting ==  "obs_transitions":
	path_prefix= "IDM-obs_"

if LfO_setting =="LfO":
	input_path = path_prefix+ dataType + "_estimated_obs_act_transitions/"
	input_fileName = "" #dataType +"_NN_estimated_observation_action_transitions_"
	network_path = path_prefix + dataType +"ImitationNetworks_LfO/"
	numFeatures = 18 
elif LfO_setting == "LfD" :
	input_path = path_prefix+ dataType + "_demonstrations/"
	input_fileName = dataType +"_observation_transitions"
	network_path = path_prefix+ dataType +"ImitationNetworks_LfD_observations/"
	numFeatures = 18


if dataType==  "4G_light":
	time_Steps = 300
elif dataType== "Flocking" :
	time_Steps = 300

def testImitationModel():
	if dataType==  "4G_light":
		evaluateLearnt4G(dimension, numberOfAgents, network_path + "imitationNetwork_" , num_repetitions, num_eval_runs , time_Steps, LfO_setting  )
	elif dataType== "Flocking" :
		evaluateLearntFlocking(dimension, numberOfAgents, network_path + "imitationNetwork_" , num_repetitions, num_eval_runs , time_Steps,LfO_setting  )


if train =="1":
	if dataType== "4G_light":
		epochs = 250 #1500
		lr = 1e-3  
	elif dataType== "Flocking":
		epochs = 250
		lr = 1e-3
	else:
		epochs = 1000
		lr = 5e-1

	SwarmTasksDataset.num_outputs = numOutputs 
	SwarmTasksDataset.num_inputs = int(numFeatures/2 )
	for i in range(num_repetitions):
		estimated_action_dataset = SwarmTasksDataset(input_path +  input_fileName + str(i)+ ".csv", 'data') #("4G_local_transitions_light_relative.csv", 'data')
		train_len = int(len(estimated_action_dataset )*0.8)
		train_set, test_set = torch.utils.data.random_split(estimated_action_dataset , [train_len, len(estimated_action_dataset ) - train_len])
		train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
		test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

		imitationNetwork = trainSupervised(train_loader , test_loader ,int(numFeatures/2)  , numOutputs, epochs, lr) 
		pickle.dump(imitationNetwork , open(network_path + "imitationNetwork_" +str(i) , 'wb'))
	testImitationModel()
else:
	testImitationModel()
