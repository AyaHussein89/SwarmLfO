from Agent import Agent
import time
import numpy as np
import copy
import math
import common as cmn
import sys


dimension = 20
numberOfAgents = 20 
dataType =   "random"   	#   # "Flocking"   #    # "4G_light" #  "random"   #  
global_logs = False
num_repetitions = 10 
storeTransitions = True # True for  LfO , False for LfD
noise_mean = 0
noise_std =  0



def setAgentParameters(numRequiredGroups, dimension,time_Steps):
	Agent.dimension= dimension
	Agent.maxTimeSteps = time_Steps

	Agent.weightOfInertia = 0
	Agent.weightCohesion =0.0
	
	Agent.weightCollision =0

	if numRequiredGroups == 1 or numRequiredGroups == 4:
		Agent.weightGoal = 0.5
	else:
		Agent.weightGoal = 0.0
	if numRequiredGroups == -1 :
		Agent.weightRand = 0.5
	else:
		Agent.weightRand = 0.0

	normaliseAgentWeights()


def normaliseAgentWeights():
	sum_weights = Agent.weightOfInertia + Agent.weightCohesion + Agent.weightCollision + Agent.weightGoal + Agent.weightRand
	Agent.weightOfInertia /= sum_weights
	Agent.weightCohesion  /= sum_weights
	Agent.weightCollision /= sum_weights
	Agent.weightGoal /= sum_weights 
	Agent.weightRand /= sum_weights



def setTargetPoints(numRequiredGroups, dimension):
	if numRequiredGroups==1:
		target_points =np.array([[dimension/2, dimension/2]])
	elif numRequiredGroups==4:
		target_points = np.array([[dimension/4,dimension/4] , [3*dimension/4,dimension/4], [3*dimension/4,3*dimension/4], [dimension/4,3*dimension/4]])
	else:
		target_points = []
	return 	target_points




def generateSheltering(numRequiredGroups, dimension, numberOfAgents, num_repetitions, eps_per_repetition , time_Steps, numSnapShots , global_logs , storeTransitions):

	setAgentParameters(numRequiredGroups, dimension,time_Steps)
	Agent.adj_file_name = "4G_light_demonstrations/" + "4G_light_state_transitions"
	Agent.file_relative_features_name = "4G_light_demonstrations/" + "4G_light_observation_transitions"	

	for i in range(num_repetitions):

		agentList = Agent(0,0, dimension , dimension ,numberOfAgents)

		agentList.init_full_tr_file(eps_per_repetition, i)
		agentList.init_raltive_file(eps_per_repetition, i)
		target_points = setTargetPoints(numRequiredGroups, dimension)
		Agent.target_points = target_points
		agentList.write_target_points(target_points)

		for e in range(eps_per_repetition):

			agentList.reset()

			for j in range(time_Steps):
				agentList.applySheltering()		
				agentList.write_absolute_local_transitions()
				agentList.write_relative_local_transitions()
					
				agentList.step += 1
				agentList.render()
				#time.sleep(0.5)
			agents_per_landmark = np.zeros(4)
			for t in range(len(target_points)):			
				for b in range(len(agentList.x)):
					if math.sqrt((agentList.x[b]-target_points[t][0])**2 + (agentList.y[b]-target_points[t][1])**2) < Agent.bodyRadius*10:
						agents_per_landmark[t]+=1 
			print(agents_per_landmark[0], "\t", agents_per_landmark[1], "\t", agents_per_landmark[2], "\t", agents_per_landmark[3], "\t")
		print("\n")
					




# for training agent-level inverse dynm model
def generateRandomLocalExp(dimension, numberOfAgents, num_repetitions, time_Steps, init_config):
	Agent.dimension= dimension
	Agent.maxTimeSteps = time_Steps
	Agent.adj_file_name = "exploration/random_local" 
	Agent.file_relative_features_name = "exploration_obs/random_local" + "_observation_transitions"

	for i in range(num_repetitions):

		agentList = Agent(0,0, dimension , dimension ,numberOfAgents)
		agentList.init_full_tr_file(len(init_config),i)
		agentList.init_raltive_file(len(init_config) ,i)

		for n in range(len(init_config)):

			agentList.x_last = np.zeros(numberOfAgents)
			agentList.y_last = np.zeros(numberOfAgents)

			if init_config[n] =="scattered":
				agentList.reset()
			elif init_config[n] =="1G":
				offset_x = np.random.uniform(1*dimension/10, 9*dimension/10)
				offset_y = np.random.uniform(1*dimension/10, 9*dimension/10)
				agentList.reset(offset_x ,offset_y , offset_x +dimension/8, offset_y+ dimension/8)
			elif init_config[n] =="4G":
				agentList.reset_fourGroup(dimension/50)

			for j in range(time_Steps):
				agentList.absolute_obs_last = copy.deepcopy(agentList.absolute_obs)
				agentList.x_last = copy.deepcopy(agentList.x)
				agentList.y_last = copy.deepcopy(agentList.y)
				agentList.relative_obs_last = copy.deepcopy(agentList.relative_obs)

		
				for k in range(numberOfAgents):
					agentList.velocityX[k] = np.random.uniform(-agentList.agentVehicleSpeedLimit,agentList.agentVehicleSpeedLimit) 
					agentList.velocityY[k] = np.random.uniform(-agentList.agentVehicleSpeedLimit,agentList.agentVehicleSpeedLimit) 
					[agentList.velocityX[k], agentList.velocityY[k]] = cmn.adjustVectorMax(np.array([agentList.velocityX[k], agentList.velocityY[k]]), agentList.agentVehicleSpeedLimit)
					agentList.orientation[k] = math.atan2(agentList.velocityY[k],agentList.velocityX[k]) # between -pi/2, pi/2
					
					agentList.x[k] = agentList.x[k] + agentList.velocityX[k]
					agentList.y[k] = agentList.y[k] + agentList.velocityY[k]
					[agentList.x[k] ,agentList.y[k]] = agentList.FixBoundaryConditionStickToBorder(agentList.x[k] ,agentList.y[k],k)								
					agentList.avoidCollisions(k)

				agentList.write_absolute_local_transitions()
				agentList.write_relative_local_transitions()
				agentList.step += 1
				#agentList.render()



def setFlockingParameters(dimension,time_Steps):
	Agent.dimension= dimension
	Agent.maxTimeSteps = time_Steps

	# Agent.agentVehicleSpeedLimit
	Agent.weightOfInertia = 0.0    
	Agent.weightCohesion = 0.1
	Agent.weightCollision = 0.1  
	Agent.weightGoal = 0.05 
	Agent.weightAlignment = 0 
	Agent.weightRand = 0.01
	normaliseAgentWeights()


def generateFlocking( dimension, numberOfAgents, num_repetitions, eps_per_repetition , time_Steps, numSnapShots , global_logs, storeTransitions):

	setFlockingParameters(dimension,time_Steps)	
	Agent.adj_file_name = "Flocking_demonstrations/" +"Flocking_state_transitions"
	Agent.file_relative_features_name = "Flocking_demonstrations/" + "Flocking_observation_transitions"
	
	print("Distance travelled \t Avg distance to neighbours \t Avg # neighbours \t Avg graph components")

	for i in range(num_repetitions):

		agentList = Agent(0,0, dimension , dimension ,numberOfAgents)
	
		agentList.init_full_tr_file(eps_per_repetition ,i)
		agentList.init_raltive_file(eps_per_repetition ,i)
		target_points =np.array([[dimension, dimension]])
		Agent.target_points = target_points
		Agent.light_intenisity =  320000 # 320000 lux @ 0.05m  == 800 lux @ 1m 

		for e in range(eps_per_repetition):
			Agent.target_points[0][0] =np.random.uniform(0, dimension)  
			Agent.target_points[0][1] = dimension
			Agent.x_landmark = Agent.target_points[:,0]
			Agent.y_landmark = Agent.target_points[:,1]

			#initialise swarm member within an square of length d/6 & with random center
			offset_x = 0 #np.random.uniform(0, 9*dimension/10)
			offset_y = 0 #np.random.uniform(0, dimension/2)	
			agentList.reset(offset_x , offset_y, offset_x + 1.5, offset_y + 1.5)
	
			#goal position used to determine flocking direction 
			for j in range(numberOfAgents):
				agentList.goal_x[j] = Agent.target_points[0][0] 
				agentList.goal_y[j] = Agent.target_points[0][1]
			init_dist = math.sqrt((np.mean(agentList.x) - Agent.target_points[0][0])**2 + (np.mean(agentList.y)- Agent.target_points[0][1])**2)
			total_neighbour_distance = 0
			total_num_negibours = 0
			total_num_graph_components = 0
	
			for j in range(time_Steps):
				agentList.updateFlockingAgentPositions(-5,-5)
				for b in range(numberOfAgents):
					[neighbour_distance_x,neighbour_distance_y, num_negibours] = agentList.getAverageNeighbourDistance(b, agentList.cohesionRange)
					total_neighbour_distance += math.sqrt(neighbour_distance_x**2+neighbour_distance_x**2)
					total_num_negibours += num_negibours 

				total_num_graph_components += len(agentList.numberOfGraphComponents())
	
				agentList.write_absolute_local_transitions()
				agentList.write_relative_local_transitions()
					
				agentList.step += 1
				#agentList.render()
			final_dist = math.sqrt((np.mean(agentList.x) - Agent.target_points[0][0])**2 + (np.mean(agentList.y)- Agent.target_points[0][1])**2)
			distance_change = final_dist - init_dist 
			print(distance_change , "\t" , total_neighbour_distance/(time_Steps*numberOfAgents) , "\t", total_num_negibours/(time_Steps*numberOfAgents), "\t" , total_num_graph_components/time_Steps, "\t",  total_num_graph_components, "\t", time_Steps )






dataType = sys.argv[1]
print(dataType)
if len(sys.argv) > 2:
	eps_per_repetition = int(sys.argv[2])
else:
	eps_per_repetition = 5

if noise_mean == 0 and noise_std == 0:
	Agent.noise_added = False
else:
	Agent.noise_added = True
	Agent.noise_mean = noise_mean
	Agent.noise_std = max(1e-5, noise_std) # to avoid std <=0 

if dataType== "random":
	time_Steps =  4000  # total exploration length = time_Steps * 2 
	numSnapShots = time_Steps
	generateRandomLocalExp(dimension, numberOfAgents, num_repetitions, time_Steps, ["scattered", "1G" ])   

elif dataType== "Flocking":
	time_Steps = 300
	#eps_per_repetition = 10 # Use 10 for performance comparison with LfD & LfO; and use 5 for demonstrations to be used for learning
	numSnapShots = time_Steps
	generateFlocking( dimension, numberOfAgents, num_repetitions, eps_per_repetition , time_Steps, numSnapShots , global_logs, storeTransitions)

elif dataType== "4G_light":
	time_Steps = 300
	#eps_per_repetition = 10 # Use 10 for performance comparison with LfD & LfO; and use 5 for demonstrations to be used for learning
	numSnapShots = time_Steps
	generateSheltering(4, dimension, numberOfAgents, num_repetitions, eps_per_repetition ,time_Steps, numSnapShots , global_logs, storeTransitions)

