import numpy as np

#for reproducibility
np.random.seed(7)
#parameters
steps=1000
particles=15000
dt=1.0
#diffusion coefficient, controls the speed of the spread
D=0.5

def brownian_motion(particles,steps,dimension,D,dt):
    std_dev=np.sqrt(2*D*dt)
    #normal distribution around 0.0 to avoid bias
    random_steps=np.random.normal(loc=0.0,scale=std_dev,size=(particles,steps,dimension))
    positions=np.zeros((particles,steps+1,dimension))
    #cumulative sum along steps (time)
    positions[:,1:,:]=np.cumsum(random_steps,axis=1)
    return positions,random_steps

pos_2d,steps_2d=brownian_motion(particles,steps,2,D,dt)
pos_3d,steps_3d=brownian_motion(particles,steps,3,D,dt)
#Euclidean distance
dist_2d=np.linalg.norm(pos_2d,axis=2)
dist_3d=np.linalg.norm(pos_3d,axis=2)
#expected value of the distance
calc_mean_dist_2d=np.mean(dist_2d,axis=0)
calc_mean_dist_3d=np.mean(dist_3d,axis=0)
t=np.arange(steps+1)*dt
#theoretical expected value of the distance using Rayleigh distribution
exp_mean_dist_2d=np.sqrt(2*D*t)*np.sqrt(np.pi/2)
#theoretical expected value of the distance using Maxwell-Boltzmann distribution
exp_mean_dist_3d=np.sqrt(2*D*t)*np.sqrt(8/np.pi)
print(f"After {steps} steps the expected and calculated distances are:")
#the last coordiante is the expected value after the last step
print(f"2D: Expected: {exp_mean_dist_2d[-1]:.6f} Calculated: {calc_mean_dist_2d[-1]:.6f}")
print(f"3D: Expected: {exp_mean_dist_3d[-1]:.6f} Calculated: {calc_mean_dist_3d[-1]:.6f}")
    
