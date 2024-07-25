import numpy as np
import common as cmn
class Shepherd:
	sheepDogVehicleSpeedLimit = 0.39

	
	def __init__(self, minX,minY, maxX, maxY):
		self.maxX=maxX
		self.maxY=maxY
		self.minX=minX
		self.minY=minY	
		
	
	def reset(self, lowX =False , lowY= False, highX=False, highY=False):
		if lowX== False:
			lowX= self.minX
			lowY= self.minY
			highX= self.maxX
			highY= self.maxY
		
		self.x = np.random.uniform(lowX,highX)
		self.y = np.random.uniform(lowY,highY)	
		self.directionX= np.random.random()
		self.directionY= np.random.random()		
		self.furthestSheepIndex = 0			
		self.timeToFindNextFurthestSheep = 0
		self.finished = False
	
	def applyShepherdAction(self, actionX, actionY):
	
		[SheepDogDirectionX, SheepDogDirectionY] = cmn.normalise(actionX,actionY)
		actionX = SheepDogDirectionX*self.sheepDogVehicleSpeedLimit
		actionY = SheepDogDirectionY*self.sheepDogVehicleSpeedLimit

		self.x = self.x + actionX
		self.y = self.y + actionY
		self.fixBoundary()


	
	def fixBoundary(self):
		if self.x < self.minX:
			self.x = self.minX
		if self.x > self.maxX:
			self.x = self.maxX			
		if self.y < self.minY:
			self.y = self.minY
		if self.y > self.maxY:
			self.y = self.maxY
	


	def updateDogPosition(self, sheep): 

		[FurthestSheepIndex, FurthestSheepDistance] = sheep.getFurthestSheepFromGoal()
		self.determineIfTaskComplete( sheep, FurthestSheepIndex)
		if self.finished:
			return
		if self.timeToFindNextFurthestSheep ==0:			
			self.furthestSheepIndex = FurthestSheepIndex
			self.timeToFindNextFurthestSheep = 30

		if np.sqrt((sheep.x[self.furthestSheepIndex]-sheep.gcm[0])**2 + (sheep.y[self.furthestSheepIndex]-sheep.gcm[1])**2) > self.neighbourhoodRange:
			targetPoint = self.calculateCollectPointToGCM(sheep)
		else:
			targetPoint = self.calculateDrivePoint(sheep)

		self.applyShepherdAction( targetPoint[0] - self.x, targetPoint[1]- self.y  )
		self.timeToFindNextFurthestSheep -=1


	

	def determineIfTaskComplete(self, sheep, FurthestSheepIndex):
		if np.sqrt((sheep.x[FurthestSheepIndex]-sheep.goal[0])**2 + (sheep.y[FurthestSheepIndex]-sheep.goal[1])**2) <= sheep.goalRadius:
			self.finished= True

	def numSheepAtGoal(self,sheep):
		atGoal = 0
		for i in range(len(sheep.x)):
			if np.sqrt((sheep.x[i]-sheep.goal[0])**2 + (sheep.y[i]-sheep.goal[1])**2) <= sheep.goalRadius:
				atGoal+=1
		return atGoal

	def calculateCollectPointToGCM(self, sheep):
		[directionFromFurthestToGcmX, directionFromFurthestToGcmY] = cmn.normalise(sheep.x[self.furthestSheepIndex] -sheep.gcm[0], sheep.y[self.furthestSheepIndex] - sheep.gcm[1])
		collectionPointX = sheep.x[self.furthestSheepIndex] + self.neighbourhoodRange * directionFromFurthestToGcmX
		collectionPointY = sheep.y[self.furthestSheepIndex] + self.neighbourhoodRange * directionFromFurthestToGcmY
		return [collectionPointX , collectionPointY ]

	def calculateCollectPointToGoal(self, sheep):
		[directionFromFurthestToGoalX, directionFromFurthestToGoalY] = cmn.normalise(sheep.x[self.furthestSheepIndex] -sheep.goal[0], sheep.y[self.furthestSheepIndex] - sheep.goal[1])
		collectionPointX = sheep.x[self.furthestSheepIndex] + self.neighbourhoodRange * directionFromFurthestToGoalX
		collectionPointY = sheep.y[self.furthestSheepIndex] + self.neighbourhoodRange * directionFromFurthestToGoalY
		return [collectionPointX , collectionPointY ]


	def calculateDrivePoint(self, sheep ):
		[directionFromGcmXToGoalX, directionFromGcmYToGoalY] = cmn.normalise(sheep.gcm[0] - sheep.goal[0] , sheep.gcm[1] - sheep.goal[1] )
		drivePointX = sheep.gcm[0]  + self.neighbourhoodRange * directionFromGcmXToGoalX
		drivePointY = sheep.gcm[1]  + self.neighbourhoodRange * directionFromGcmYToGoalY
		return [drivePointX , drivePointY]