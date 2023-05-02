import numpy as np
import sys


dataType = sys.argv[1] # can be  "random" ,  "4G_light"  or "Flocking" 
num_repetitions = 10

# LFO data
def readLfODataLocal(input_path, output_path, file_name, numTargetPoints, num_features = 4, num_outputs = 2):
	infile = open(input_path+file_name, 'r')
	num_boids = int(infile.readline())
	time_steps = int(infile.readline())
	num_repetitions = int(infile.readline())
		

	features = np.zeros((time_steps*num_repetitions , num_boids, num_features  ))
	outputs = np.zeros((time_steps*num_repetitions , num_boids, num_outputs   ))
	
	for i in range(numTargetPoints*2):
		line = infile.readline()

	for i in range(time_steps*num_repetitions ):
		for j in range(num_boids):
			for k in range(num_features):
				line = infile.readline()
				try:
					features[i][j][k] = float(line)
				except:
					print("error when reading: ", line)
					exit()

			for k in range(num_outputs):
				line = infile.readline()
				try:
					outputs[i][j][k] = float(line)
				except:
					print("error when reading: ",line)
					exit()
		
	infile.close()
	with open(output_path + file_name + ".csv", "w") as f:
		for i in range(time_steps*num_repetitions ):
			np.savetxt(f,np.concatenate((features[i] , outputs[i]), axis=1) , fmt = '%0.5f' ,delimiter=",")
	return features , outputs







if dataType == "random":
	input_path = "exploration/"
	input_path_obs = "exploration_obs/"
	output_path = "exploration/"
	output_path_obs = "exploration_obs/"
	for i in range(num_repetitions):
		readLfODataLocal(input_path, output_path, "random_local" +str(i), 0, 10 ,2)
		readLfODataLocal(input_path_obs, output_path_obs, "random_local_observation_transitions" +str(i), 0,18,2)

elif dataType == "Flocking" :
	input_path = "Flocking_demonstrations/"
	input_path_obs = "Flocking_demonstrations/"
	output_path = "Flocking_demonstrations/"
	output_path_obs = "Flocking_demonstrations/"

	for i in range(num_repetitions):
		readLfODataLocal(input_path, output_path, "Flocking_state_transitions" +str(i), 0,10,2)
		readLfODataLocal(input_path_obs, output_path_obs, "Flocking_observation_transitions"+str(i), 0,18,2)


elif dataType == "4G_light" :
	input_path = "4G_light_demonstrations/"
	input_path_obs = "4G_light_demonstrations/"
	output_path = "4G_light_demonstrations/"
	output_path_obs = "4G_light_demonstrations/"
	for i in range(num_repetitions):
		readLfODataLocal(input_path, output_path, "4G_light_state_transitions" +str(i), 4, 10 , 2)
		readLfODataLocal(input_path_obs, output_path_obs, "4G_light_observation_transitions"+str(i), 0, 18, 2)



