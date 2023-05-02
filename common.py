import numpy as np

# v is 2D vector	
def magnitude( x,y=False):
	if y==False:
		return np.sqrt(np.sum(x**2))
	return np.sqrt(x**2 + y**2)
	
# v is 2D vector	
#def magnitude(v):
	#return np.sqrt(v[0]**2 + v[1]**2)

	
# v1 & v2 are 2D vectors		
def distanceBetween( v1, v2):
	return np.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
	
	
# v is 2D vector	
def normalise( v):
	magnitude= np.sqrt(np.sum(v**2))
	if magnitude> 0:
		return v/magnitude
	return v	
	
	
# v is 2D vector	
def adjustVectorMax( v, maxVal):
	magnitude= np.sqrt(np.sum(v**2))
	if magnitude<= maxVal:
		return v		
	return v/magnitude * maxVal 
	
	
# individual components of a vector	
def normalise( x, y):
	magnitude= np.sqrt(x**2+ y**2)
	if magnitude> 0:
		return [x/magnitude, y/magnitude]
	return [x,y]
	
#https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
#https://www.semanticscholar.org/paper/Approximating-the-Kullback-Leibler-Divergence-Hershey-Olsen/831780b12cb41a9905c3d4f58831a2ea6d09223b?p2df
def KLDivergence(u1, u2, sigma1, sigma2):
	return (np.log(sigma2/sigma1) + (sigma1**2 +(u1-u2)**2)/(2*(sigma2)**2) -0.5)