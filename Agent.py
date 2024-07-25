import numpy as np
import common as cmn
import cv2 
from tkinter import *
import time
import math
import random
import sys
from Graph import Graph
import copy
import csv

class Agent:

	maxTimeSteps = 50
	numberOfAgent= 20
	dimension=0	
	goal=[10,10]
	isDispersion = False

	#Physical specs similar to  e-puck version 1.3 (All dimensions are in meters)
	# see https://www.gctronic.com/doc/index.php/e-puck2
	bodyRadius = 0.035  
	agentVehicleSpeedLimit = 0.13
	wheelRadius = 0.021
	distBetweenWheels = 0.053


	# algorithmic parameters
	weightCohesion  =  0.5
	weightCollision = 1.05
	weightAlignment = 0.0
	weightInfluenceOfDog = 0
	weightOfInertia = 0.2 				
	weightRand= 0.1
	weightGoal = 0
	# the following are parameters set within 6 cm (epuck has distance sensor measuring ambient light and proximity of objects up to 6 cm)
	collisionRange = bodyRadius*3 
	sensingRange=  collisionRange*5
	cohesionRange =  sensingRange
	alignmentRange=  sensingRange 
	influenceRange = sensingRange *3

	x_landmark = None
	y_landmark = None
	target_points = [[0,0]]


	iconDimension =30
	
	light_intenisity = 1000 # lux @ 0.05m    (2.5 lumens)
	light_sensitivity_min = 1 #in lux
	light_sensitivity_max = 10000 #in lux
	adj_file_name= ""
	noise_added = False
	noise_std= 0
	noise_mean = 0
	
	def __init__(self, minX,minY, maxX, maxY,numberOfAgent):
		self.numberOfAgent= numberOfAgent
		self.maxX=maxX
		self.maxY=maxY
		self.minX=minX
		self.minY=minY	

		self.x = np.random.uniform(minX,maxX,numberOfAgent)
		self.y = np.random.uniform(minY,maxY,numberOfAgent)
		self.goal_x=np.array([self.goal[0] for i in range(numberOfAgent)])
		self.goal_y=np.array([self.goal[1] for i in range(numberOfAgent)])
		self.orientation = np.random.uniform(-math.pi,math.pi,numberOfAgent)

		self.neighbours = np.zeros(numberOfAgent) # num neighbours for each agent

		self.velocityX= np.zeros(numberOfAgent)
		self.velocityY= np.zeros(numberOfAgent)
		self.velocityX_last= np.zeros(numberOfAgent)
		self.velocityY_last= np.zeros(numberOfAgent)


		#rendering
		self.relative_observation_shape = (650, 650, 3)
		self.factor= np.uint8((650-self.iconDimension*2)/self.dimension) 
		self.canvas = np.ones(self.relative_observation_shape, dtype=np.uint8) * 255   

		self.agentIcon = cv2.imread("agent.jpg") #/ 255.0
		self.agentIcon_w = 4 
		self.agentIcon_h = 4 
		self.agentIcon = cv2.resize(self.agentIcon, (self.agentIcon_h, self.agentIcon_w))
		
		self.landmarkIcon =  cv2.imread("landmark.jpg") 
		self.landmarkIcon_w =  15 
		self.landmarkIcon_h = 15 
		self.landmarkIcon = cv2.resize(self.landmarkIcon, (self.landmarkIcon_h, self.landmarkIcon_w))

		self.dogIcon = cv2.imread("dog.jpg") 
		self.dogIcon_w = 6   
		self.dogIcon_h = 6   
		self.dogIcon = cv2.resize(self.dogIcon, (self.dogIcon_h, self.dogIcon_w))

			

	def reset(self, lowX =False , lowY= False, highX=False, highY=False):

		if lowX== False and highX == False:
			lowX= self.minX
			lowY= self.minY
			highX= self.maxX
			highY= self.maxY

		self.x = np.random.uniform(lowX,highX,self.numberOfAgent)
		self.y = np.random.uniform(lowY,highY,self.numberOfAgent)
		for i in range(self.numberOfAgent):
			while not self.isSufficientlyFar(i):
				self.x[i] = np.random.uniform(lowX,highX)
				self.y[i] = np.random.uniform(lowY,highY)

		self.orientation = np.random.uniform(-math.pi,math.pi,self.numberOfAgent)
		self.recent_collisionX = np.zeros(self.numberOfAgent)
		self.recent_collisionY = np.zeros(self.numberOfAgent)
	
		self.velocityX= np.zeros(self.numberOfAgent)
		self.velocityY= np.zeros(self.numberOfAgent)
		self.velocityX_last= np.zeros(self.numberOfAgent)
		self.velocityY_last= np.zeros(self.numberOfAgent)

		self.relative_obs = []
		for i in range(self.numberOfAgent):
			self.relative_obs.append(self.get_relative_obs(i))

		self.absolute_obs = []
		for i in range(self.numberOfAgent):
			self.absolute_obs.append(self.get_absolute_obs(i))
		

		self.step=0



	# before initialising an agent's position, check make sure it is not too close to any of the previously initialised agents
	def isSufficientlyFar(self,agent_id):
		for i in range(agent_id):
			temp_x = abs(self.x[agent_id] - self.x[i])
			temp_y = abs(self.y[agent_id] - self.y[i])
			if temp_x < Agent.bodyRadius*2  and temp_y < Agent.bodyRadius*2:
				return False
		return True

		
	def init_full_tr_file(self,num_eps,file_number):
		self.file_adj_features  = open( self.adj_file_name +str(file_number), "w") 
		self.file_adj_features.write("%1.0f\n" % self.numberOfAgent)
		self.file_adj_features.write("%1.0f\n" % self.maxTimeSteps)
		self.file_adj_features.write("%1.0f\n" % num_eps) 
		self.csv_file_abs = self.adj_file_name +str(file_number) + ".csv"
		with open(self.csv_file_abs, mode='w') as file:
			pass		

	def init_raltive_file(self,num_eps, file_number):
		self.file_relative_features  = open( self.file_relative_features_name +str(file_number) , "w") 
		self.file_relative_features.write("%1.0f\n" % self.numberOfAgent)
		self.file_relative_features.write("%1.0f\n" % self.maxTimeSteps)
		self.file_relative_features.write("%1.0f\n" % num_eps) 
		self.csv_file_relative = self.file_relative_features_name +str(file_number) + ".csv"
		with open(self.csv_file_relative, mode='w') as file:
			pass


	# Expert sheltering algorithm
	def applySheltering(self): 
		self.relative_obs_last = copy.deepcopy(self.relative_obs)
		self.absolute_obs_last= copy.deepcopy(self.absolute_obs)
		self.x_last = copy.deepcopy(self.x)
		self.y_last = copy.deepcopy(self.y)
		for i in range(self.numberOfAgent):
			self.velocityX[i] = 0	
			self.velocityY[i] = 0

			if self.recent_collisionX[i]==1 or self.recent_collisionY[i]==1: # rotate if recent collision
				self.velocityX[i] = random.uniform(-self.agentVehicleSpeedLimit,self.agentVehicleSpeedLimit) 
				self.velocityY[i] = random.uniform(-self.agentVehicleSpeedLimit,self.agentVehicleSpeedLimit) 
				[self.velocityX[i], self.velocityY[i]] = cmn.adjustVectorMax(np.array([self.velocityX[i], self.velocityY[i]]), self.agentVehicleSpeedLimit)
				self.orientation[i] = math.atan2(self.velocityY[i],self.velocityX[i]) # between -pi, pi
				
			elif self.relative_obs[i][4] == 0: # move forward if no collisions or light detected
				self.velocityX[i] = math.cos(self.orientation[i]) * self.agentVehicleSpeedLimit
				self.velocityY[i] = math.sin(self.orientation[i]) * self.agentVehicleSpeedLimit

			else: # if light detected, move along its direction
				self.orientation[i] = self.relative_obs[i][5] 
				self.velocityX[i] = math.cos(self.orientation[i]) * self.agentVehicleSpeedLimit
				self.velocityY[i] = math.sin(self.orientation[i]) * self.agentVehicleSpeedLimit	

			self.x[i] = self.x[i] + self.velocityX[i]
			self.y[i] = self.y[i] + self.velocityY[i]
			[self.x[i] ,self.y[i]] = self.FixBoundaryConditionStickToBorder(self.x[i] ,self.y[i],i)			
			self.avoidCollisions(i)
		

			

	# takes dog position and calls the functions for calculating the new position of each agent; set as -5 if no dog exist in the environment
	def updateFlockingAgentPositions(self, dogX = -5, dogY =-5):
		self.dog_x = dogX
		self.dog_y = dogY
		self.step +=1
		NewLocationX= np.zeros(self.numberOfAgent)
		NewLocationY= np.zeros(self.numberOfAgent)
		NewLocationVelX= np.zeros(self.numberOfAgent)
		NewLocationVelY= np.zeros(self.numberOfAgent)
		self.x_last = copy.deepcopy(self.x)
		self.y_last = copy.deepcopy(self.y)
		self.relative_obs_last = copy.deepcopy(self.relative_obs)
		self.absolute_obs_last = copy.deepcopy(self.absolute_obs)


		for i in range(self.numberOfAgent):
			self.velocityX_last[i]= self.velocityX[i]
			self.velocityY_last[i]= self.velocityY[i]

			NewLocation = self.agentflocking(dogX, dogY,i)
			MNewLocation = self.FixBoundaryConditionStickToBorder(NewLocation[0],NewLocation[1],i)
			NewLocationX[i]=MNewLocation[0]
			NewLocationY[i]=MNewLocation[1]
			NewLocationVelX[i]=NewLocation[2]
			NewLocationVelY[i]=NewLocation[3]
			self.neighbours[i] = NewLocation[4]
	
		for i in range(self.numberOfAgent):
			self.x[i] = NewLocationX[i]
			self.y[i] = NewLocationY[i]
			self.avoidCollisions(i) 
			self.velocityX[i] = NewLocationVelX[i]
			self.velocityY[i] = NewLocationVelY[i]
			self.orientation[i] = math.atan2(self.velocityY[i],self.velocityX[i]) 		




	# logging target points in the environment, if any
	def write_target_points(self, target_points): 
		if len(target_points) >0:
			self.x_landmark = target_points[:,0]
			self.y_landmark = target_points[:,1]
		for i in range(len(target_points)):
			self.file_adj_features.write("%1.2f\n"%(target_points[i][0]))
			self.file_adj_features.write("%1.2f\n"%(target_points[i][1]))


	
	# logs agent state transitions in the format: 	s(t), s(t+1), a(t)
	def write_absolute_local_transitions(self):
		with open(self.csv_file_abs, mode='a', newline='') as file:
			writer = csv.writer(file)
			for i in range(len(self.x)):
				self.absolute_obs[i] = self.get_absolute_obs(i)
				writer.writerow(self.absolute_obs_last[i] + self.absolute_obs[i] + [self.velocityX[i], self.velocityY[i]])



	# logs agent observation transitions in the format: 	o(t), o(t+1), a(t)
	def write_relative_local_transitions(self):
		with open(self.csv_file_relative, mode='a', newline='') as file:
			writer = csv.writer(file)
			for i in range(len(self.x)):
				self.relative_obs[i] = self.get_relative_obs(i)
				writer.writerow(self.relative_obs_last[i] + self.relative_obs[i] + [self.velocityX[i], self.velocityY[i]])



	# observations
	def get_relative_obs(self,i):		
		local_obs = []
		vec = self.getAverageNeighbourDistance(i, self.collisionRange)
		local_obs.append((vec[0])/self.dimension )
		local_obs.append((vec[1])/self.dimension )
		vec = self.getAverageNeighbourDistance(i, self.cohesionRange)
		local_obs.append((vec[0])/self.dimension )
		local_obs.append((vec[1])/self.dimension )
		
		sensed_light, delta_x, delta_y  = self.estimate_light_val_orientation(i)
		local_obs.append(sensed_light )
		local_obs.append(math.atan2(delta_y, delta_x))

		local_obs.append(self.orientation[i])
		local_obs.append(self.recent_collisionX[i])
		local_obs.append(self.recent_collisionY[i])
		return local_obs



	# states
	def get_absolute_obs(self,i):
		if Agent.noise_added:
			noise = np.random.normal(Agent.noise_mean, Agent.noise_std, 3)
		else:
			noise = np.zeros(3)
		local_obs = []
		local_obs.append((self.x[i]+ noise[0])/self.dimension )
		local_obs.append((self.y[i]+ noise[1])/self.dimension )
		local_obs.append(self.orientation[i]+ noise[2]/math.pi)
		local_obs.append(self.recent_collisionX[i])
		local_obs.append(self.recent_collisionY[i])

		return local_obs



	def estimate_light_val_orientation(self, i):
		nearest_light_id = -1
		nearest_light_dist_sqrd = (self.dimension*2) **2

		for j in range(len(self.target_points)):
			dist_sqrd = (self.x[i]-self.target_points[j][0])**2 + (self.y[i]-self.target_points[j][1])**2
			if dist_sqrd < nearest_light_dist_sqrd :
				nearest_light_dist_sqrd = dist_sqrd 
				nearest_light_id =  j
		
		sensed_intensity= min(Agent.light_intenisity/(nearest_light_dist_sqrd+0.00001), self.light_sensitivity_max)
		if sensed_intensity> self.light_sensitivity_min:
			# first value is sensed_intensity/self.light_sensitivity_max to be normalised
			return sensed_intensity/self.light_sensitivity_max, (self.target_points[nearest_light_id][0] - self.x[i])/self.dimension, (self.target_points[nearest_light_id][1] - self.y[i])/self.dimension
			
		else:

			return 0, 0, 0



	# calculate aberage delta x and delta y from neighbours within a given range
	def getAverageNeighbourDistance(self, agentID, range):
		distances = self.agentDistancesToAAgent(agentID)
		neighIndices =[i for i,v in enumerate(distances) if v <range]
		numberOfNeigh = len(neighIndices ) - 1
		
		if numberOfNeigh > 0:			
			neighX = (self.x[agentID] - self.x[neighIndices]).sum() / numberOfNeigh 
			neighY = (self.y[agentID] - self.y[neighIndices]).sum() / numberOfNeigh 
		else:
			neighX =0
			neighY =0

		return neighX , neighY, numberOfNeigh 


	# dog information is relevant only in shepherding tasks
	def agentflocking(self,dogX, dogY,AgentID):

		[GCMAlignmentorientationX, GCMAlignmentorientationY,GCMorientationX, GCMorientationY,CollisionX, CollisionY,numNeighbours] = self.flockingVelocities(AgentID)

		[GCMAlignmentorientationX,GCMAlignmentorientationY]= cmn.normalise(GCMAlignmentorientationX,GCMAlignmentorientationY)		
		[GCMorientationX,GCMorientationY]= cmn.normalise(GCMorientationX,GCMorientationY)		
		[CollisionX,CollisionY] =cmn.normalise(CollisionX,CollisionY)

		goalorientationX = self.goal_x[AgentID] - self.x[AgentID]
		goalorientationY = self.goal_y[AgentID] - self.y[AgentID]
		[goalorientationX, goalorientationY] = cmn.normalise(goalorientationX ,goalorientationY) 

		oldvelocityX = self.velocityX[AgentID]
		oldvelocityY = self.velocityY[AgentID]

		velocityRequiredX =  Agent.weightCohesion * GCMorientationX + Agent.weightAlignment * GCMAlignmentorientationX + Agent.weightCollision * CollisionX + Agent.weightGoal * goalorientationX + Agent.weightRand * random.uniform(-1, 1)
		velocityRequiredY =  Agent.weightCohesion * GCMorientationY + Agent.weightAlignment * GCMAlignmentorientationY + Agent.weightCollision * CollisionY + Agent.weightGoal * goalorientationY +Agent.weightRand * random.uniform(-1, 1)

		if Agent.isDispersion and velocityRequiredX ==0 and velocityRequiredY == 0 :
			velocityRequiredX = math.cos(self.orientation[AgentID] )
			velocityRequiredY = math.sin(self.orientation[AgentID] )
		else:
			velocityRequiredX +=  Agent.weightOfInertia*oldvelocityX
			velocityRequiredY +=  Agent.weightOfInertia*oldvelocityY
			

		#relevant only for shepherding tasks
		CurrentPosition = [self.x[AgentID], self.y[AgentID]]
		orientationToDogX =  CurrentPosition[0] - dogX
		orientationToDogY =  CurrentPosition[1] - dogY
		orientationToDog = np.array([orientationToDogX,orientationToDogY]) 
		orientationToDogMagnitude=np.sqrt(np.sum(orientationToDog**2))
		if  orientationToDogMagnitude < self.influenceRange:
			orientationToDog=orientationToDog/orientationToDogMagnitude
			velocityRequiredX += Agent.weightInfluenceOfDog * orientationToDog[0] *(1- orientationToDogMagnitude/self.influenceRange)**2
			velocityRequiredY += Agent.weightInfluenceOfDog * orientationToDog[1] *(1- orientationToDogMagnitude/self.influenceRange)**2			
			

				

    
		XNew = CurrentPosition[0] + velocityRequiredX
		YNew = CurrentPosition[1] + velocityRequiredY

		if XNew < self.minX:
			velocityRequiredX = 0.0 - velocityRequiredX
		
		if XNew > self.maxX:
			velocityRequiredX = 0.0 - velocityRequiredX
		
		if YNew < self.minY:
			velocityRequiredY = 0.0 - velocityRequiredY
		
		if YNew > self.maxY:
			velocityRequiredY = 0.0 - velocityRequiredY
			

		VelocityCurrent = np.array([velocityRequiredX, velocityRequiredY])
		Output=self.agentvehiclemodel(CurrentPosition, VelocityCurrent, np.array([velocityRequiredX, velocityRequiredY]))

		AgentorientationX =  Output[0][0]
		AgentorientationY =  Output[0][1]
		Xnew = Output[1][0]
		Ynew =  Output[1][1]


		return [Xnew,Ynew,AgentorientationX,AgentorientationY,numNeighbours]

		

	def flockingVelocities(self, agentID):
		
		GCMAlignmentorientationX=0
		GCMAlignmentorientationY=0
		GCMorientationX=0
		GCMorientationY=0
		RepulsionX  = 0
		RepulsionY  = 0
		
		distances = self.agentDistancesToAAgent(agentID)
		
		cohesionIndices = [i for i,v in enumerate(distances) if v <self.cohesionRange]
		numberOfCohesion = len(cohesionIndices)-1
		#Calculating the local centre of mass for all agent within the cohesion range of that agent and the orientation between the agent of interest and 
		#the global centre of mass of the agent in its neighbourhood
		if numberOfCohesion > 0:
			GCMorientationX = (self.x[cohesionIndices].sum() -  self.x[agentID]) / numberOfCohesion
			GCMorientationY = (self.y[cohesionIndices].sum() -  self.y[agentID]) / numberOfCohesion
			GCMorientationX = GCMorientationX -  self.x[agentID]
			GCMorientationY = GCMorientationY -  self.y[agentID]
		

		#Calculating the average alignment vector then calculating the alignment vector between the velocity of the agent of interest 
		#and the average orientation of the agent in its neighbourhood   
		alignmentIndices = [i for i,v in enumerate(distances) if v <self.alignmentRange]
		numberOfAllignment = len(alignmentIndices) -1 
		if numberOfAllignment > 0:
			GCMAlignmentorientationX = (self.velocityX[cohesionIndices].sum() -  self.velocityX[agentID] ) / numberOfAllignment
			GCMAlignmentorientationY = (self.velocityY[cohesionIndices].sum() -  self.velocityY[agentID] ) / numberOfAllignment
			GCMAlignmentorientationX = GCMAlignmentorientationX -   self.velocityX[agentID]
			GCMAlignmentorientationY = GCMAlignmentorientationY -   self.velocityY[agentID]
		

		#Calculating the average orientations for collision avoidance
		repulsionIndices =[i for i,v in enumerate(distances) if v <self.collisionRange]
		numberOfRepulsion = len(repulsionIndices) - 1		
		if numberOfRepulsion > 0:			
			RepulsionX = (self.x[agentID] - self.x[repulsionIndices]).sum() / numberOfRepulsion
			RepulsionY = (self.y[agentID] - self.y[repulsionIndices]).sum() / numberOfRepulsion


		return [GCMAlignmentorientationX,GCMAlignmentorientationY,GCMorientationX,GCMorientationY,RepulsionX,RepulsionY,numberOfCohesion]
		 
	

	# to be used in the future to simulate a specific phyics model of the agent. 	
	def agentvehiclemodel(self, Position, VelocityCurrent, VelocityRequired):
		VelocityNew = cmn.adjustVectorMax(VelocityRequired, self.agentVehicleSpeedLimit)
		return [VelocityNew, Position + VelocityNew]
		



	def avoidCollisions(self, agentID):
		distances = self.lastAgentDistancesToAAgent(agentID)

		neighIndices =[i for i,v in enumerate(distances) if v <Agent.bodyRadius*2] #min dist
		numberOfNeigh = len(neighIndices ) - 1


		for i in neighIndices:
			if i == agentID:
				continue
			if abs(self.x[agentID]- self.x_last[i]) < abs(self.x_last[agentID]- self.x_last[i]) and abs(self.x[agentID]- self.x_last[i]) < Agent.bodyRadius*2:
				self.x[agentID] = copy.deepcopy(self.x_last[agentID])
				self.recent_collisionX[agentID] = 1

			if abs(self.y[agentID]- self.y_last[i]) < abs(self.y_last[agentID]- self.y_last[i]) and abs(self.y[agentID]== self.y_last[i]) < Agent.bodyRadius*2:
				self.y[agentID] = copy.deepcopy(self.y_last[agentID])
				self.recent_collisionY[agentID] = 1	




	def adjustAnglePi(self, i):
		while self.orientation[i]> math.pi:
			self.orientation[i]-= math.pi*2
		while self.orientation[i] < -math.pi:
			self.orientation[i]+= math.pi*2


	def FixBoundaryConditionStickToBorder(self,PositionX,PositionY, i):
		self.recent_collisionX[i] = 0
		self.recent_collisionY[i] = 0
		if PositionX < self.minX: 
			PositionX = self.minX
			self.recent_collisionX[i] = 1
		if PositionY < self.minY: 
			PositionY = self.minY
			self.recent_collisionY[i] = 1
		if PositionX > self.maxX: 
			PositionX = self.maxX 
			self.recent_collisionX[i] = 1
		if PositionY > self.maxY: 
			PositionY = self.maxY
			self.recent_collisionY[i] = 1

		return np.array([PositionX,PositionY])					
		
		
	# distance Between all agent and a specific agent i
	def agentDistancesToAAgent(self,i):
		distances =[0 for k in range(self.numberOfAgent)]
		diffX=  self.x - self.x[i]
		diffY=  self.y - self.y[i]
		for i in range(self.numberOfAgent):
			distances[i] = np.sqrt(diffX[i]**2 +diffY[i]**2)
		return distances 

	# distance Between all agent and a specific agent i at last timestep			
	def lastAgentDistancesToAAgent(self,i):
		distances =[0 for k in range(self.numberOfAgent)]
		diffX=  self.x_last - self.x_last[i]
		diffY=  self.y_last - self.y_last[i]
		for i in range(self.numberOfAgent):
			distances[i] = np.sqrt(diffX[i]**2 +diffY[i]**2)
		return distances 	


	def calculateGCM(self, excludeAtGoal):
		if excludeAtGoal:
			self.gcm= [0,0]
			num=0
			for i in range(self.numberOfAgent):
				if cmn.magnitude(self.x[i]-self.goal[0],self.y[i]-self.goal[1])> self.goalRadius:
					num+=1
					self.gcm[0] += self.x[i]
					self.gcm[1] += self.y[i]
			if num>0:	
				self.gcm[0]/= num
				self.gcm[1]/= num
			
		else:
			self.gcm = [np.mean(self.x) , np.mean(self.y)]	



	# get index of furthest sheep
	def getFurthestSheepFromGCM(self, excludeAtGoal=False):
		FurthestSheepDistance = 0
		FurthestSheepIndex = -1
		self.calculateGCM(excludeAtGoal)
		
		X = self.x - self.gcm[0]
		Y = self.y - self.gcm[1]
		Distances = np.sqrt(X**2 + Y**2)
			
		for i in range(self.numberOfAgent):
			if Distances[i] > FurthestSheepDistance and (excludeAtGoal and cmn.distanceBetween(self.goal, np.array([self.x[i],self.y[i]]))> self.goalRadius or not excludeAtGoal):
				FurthestSheepDistance = Distances[i]
				FurthestSheepIndex = i           		    
		return [FurthestSheepIndex, FurthestSheepDistance]

	def getFurthestSheepFromGoal(self):
		FurthestSheepDistance = 0
		FurthestSheepIndex = -1
		self.calculateGCM(excludeAtGoal=False)
		
		X = self.x - self.goal[0]
		Y = self.y - self.goal[1]
		Distances = np.sqrt(X**2 + Y**2)
			
		for i in range(self.numberOfAgent):
			if Distances[i] > FurthestSheepDistance :
				FurthestSheepDistance = Distances[i]
				FurthestSheepIndex = i           		    
		return [FurthestSheepIndex, FurthestSheepDistance]

		
			

	def numberOfGraphComponents(self):
		graph= Graph(self.numberOfAgent)
		for i in range(self.numberOfAgent):
			X = self.x - self.x[i]
			Y = self.y - self.y[i]
			Distances = np.sqrt(X**2 + Y**2)
			for j in range(self.numberOfAgent):	
				if Distances[j]<self.sensingRange:
					graph.addEdge(i,j)


		return 	graph.connectedComponents()


	def render(self, mode = "human"):
		self.draw_elements_on_canvas()
		if self.canvas.dtype != np.uint8:
			self.canvas = self.canvas.astype(np.uint8)
		if mode == "human":
			cv2.imshow("Game", self.canvas)
			cv2.waitKey(10)
    
		elif mode == "video":
			self.video_writer.write(self.canvas)

			
	def draw_elements_on_canvas(self):		

		self.canvas.fill(255)

		if not (self.x_landmark is None):
			elem_shape = self.landmarkIcon.shape
			for i in range(len(self.x_landmark)):
				x,y = int(self.x_landmark[i]*self.factor+self.iconDimension), int(self.y_landmark[i]*self.factor+self.iconDimension)
				self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = self.landmarkIcon

		elem_shape = self.agentIcon.shape
		for i in range(self.numberOfAgent):
			x,y = int(self.x[i]*self.factor+self.iconDimension), int(self.y[i]*self.factor+self.iconDimension)
			self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = self.agentIcon
		
		if (self.dog_x >=0 and self.dog_y >=0 and self.dog_x <=self.dimension and self.dog_y <= self.dimension):
			elem_shape = self.dogIcon.shape
			x,y = int(self.dog_x*self.factor+self.iconDimension), int(self.dog_y*self.factor+self.iconDimension)
			self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = self.dogIcon
		
		text = 'Time: {}'.format(self.step)
		self.canvas = cv2.putText(self.canvas, text, (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 1, cv2.LINE_AA)


	def initialise_video(self, output_file, fps =30, width = 650, height=  650):
		self.video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (width ,height))

	def finish_recording(self):
		self.video_writer.release()
		cv2.destroyAllWindows()

