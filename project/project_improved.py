import numpy as np
import matplotlib.pyplot as plt

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
#calculate the deviation of the simulation from theoretical predictions
error_2d = np.abs(calc_mean_dist_2d - exp_mean_dist_2d)
error_3d = np.abs(calc_mean_dist_3d - exp_mean_dist_3d)
print("Final 2D Error:", error_2d[-1])
print("Final 3D Error:", error_3d[-1])

#plot of the distance and the expected value
plt.figure(figsize=(10,6))
plt.plot(t, calc_mean_dist_2d, label='2D Simulated', color='blue')
plt.plot(t, exp_mean_dist_2d, '--', label='2D Theory', color='cyan')
plt.plot(t, calc_mean_dist_3d, label='3D Simulated', color='red')
plt.plot(t, exp_mean_dist_3d,'--', label='3D Theory', color='orange')
plt.xlabel('Time')
plt.ylabel('Mean Distance from Origin')
plt.title('Expected Distance vs Time')
plt.legend()
plt.grid(True)
plt.show()

#number of paths that we plot
num_paths = 20
#plot of the 2D Brownian motion
plt.figure(figsize=(8,8))
#plotting each path
for i in range(num_paths):
    plt.plot(pos_2d[i,:,0], pos_2d[i,:,1], alpha=0.7)
plt.scatter(0,0,color='red',label='Origin')
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Brownian Motion Trajectories")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

#addition to use the 3D toolkit
from mpl_toolkits.mplot3d import Axes3D
#plot of the 3D Brownian motion
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
#number of paths that we plot
num_paths = 10
#plotting each path
for i in range(num_paths):
    ax.plot(pos_3d[i,:,0], pos_3d[i,:,1], pos_3d[i,:,2])
ax.scatter(0,0,0,color='red',s=100,label='Origin')
ax.set_title("3D Brownian Motion")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.show()

#final position heatmap
#final coordinates after all steps
x_final = pos_2d[:, -1, 0]
y_final = pos_2d[:, -1, 1]
#density heatmap
plt.figure(figsize=(8,6))
plt.hist2d(x_final, y_final, bins=100, cmap='inferno')
plt.colorbar(label='Particle Count')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Probability Density after 1000 Steps')
plt.show()

#contour map, shows the final distribution
plt.figure(figsize=(8,6))
hist = plt.hist2d(x_final, y_final, bins=100, density=True, cmap='viridis')
#converting the final positions inta a probability density
H,xedges,yedges = np.histogram2d(x_final, y_final, bins=60, density=True)
#getting the center coordinate for each bin
X,Y = np.meshgrid((xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2)
plt.contour(X, Y, H.T, levels=8, colors='white')
plt.colorbar(hist[3], label='Probability Density')
plt.title("Density Contours of Final Positions")
plt.show()

#time evolution heatmaps, shows distribution over time
times = [50,100,500,1000]
#creating 4 plots
fig,ax = plt.subplots(2,2,figsize=(12,10))
#building a density map for each time snapshot
for axis,timestep in zip(ax.ravel(),times):
    axis.hist2d(pos_2d[:,timestep,0], pos_2d[:,timestep,1], bins=60, cmap='viridis')
    axis.set_title(f"t = {timestep}")
plt.tight_layout()
plt.show()

#addition to make short animations
from matplotlib.animation import FuncAnimation
#animating induvidual particle movements
fig,ax = plt.subplots(figsize=(8,8))
#creating an empty scatter plot, each dot is one particle
scatter = ax.scatter([],[],s=2)
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
ax.set_title("2D Brownian Motion")
#funtion that tells what to draw for each animation frame
def update(frame):
    scatter.set_offsets(pos_2d[:,frame,:])
    return scatter,
#frames make the animation fastern interval is miliseconds between frames, blit only redraws the particles that have moved (great for efficiency)
ani = FuncAnimation(fig, update, frames=range(0,steps,10), interval=50, blit=True)
plt.show()

from scipy import stats, optimize, integrate, ndimage

dist_f_2d=dist_2d[:,-1]
dist_f_3d=dist_3d[:,-1]

#PDF of the distance in 2D and 3D
def rayleigh_pdf(r,std_dev):
    return (r/std_dev**2)*np.exp(-r**2/(2*std_dev**2))

def maxwell_pdf(r,std_dev):
    return np.sqrt(2/np.pi)*(r**2/std_dev**3)*np.exp(-r**2/(2*std_dev**2))

t_total=steps*dt
std_dev_theory=np.sqrt(2*D*t_total)
fig,axes=plt.subplots(1,2,figsize=(15,6))
for idx,(r_data,dim_label,pdf_func,ax) in enumerate([(dist_f_2d,"2D (Rayleigh)",rayleigh_pdf,axes[0]),(dist_f_3d,"3D (Maxwell-Boltzmann)",maxwell_pdf,axes[1])]):
    counts,bin_edges,filler=ax.hist(r_data,bins=60,density=True,alpha=0.4,color='steelblue',label='Simulated histogram')
    bin_centers=0.5*(bin_edges[:-1]+bin_edges[1:])
    r_grid=np.linspace(0,np.max(r_data)*1.05,500)
    #KDE
    kde=stats.gaussian_kde(r_data,bw_method='scott')
    kde_vals=kde(r_grid)
    #least squares curve fitting
    popt,pcov=optimize.curve_fit(pdf_func,bin_centers,counts,p0=[std_dev_theory])
    sigma_fit=popt[0]
    sigma_err=np.sqrt(np.diag(pcov))[0]
    fit_vals=pdf_func(r_grid,sigma_fit)
    theory_vals=pdf_func(r_grid,std_dev_theory)
    #checking the area under the PDF
    area_kde=integrate.simpson(kde_vals,x=r_grid)
    area_fit=integrate.simpson(fit_vals,x=r_grid)
    #calculating the coefficient of determination
    y_pred_fit=pdf_func(bin_centers,sigma_fit)
    y_pred_kde=kde(bin_centers)
    ss_res_fit=np.sum((counts-y_pred_fit)**2)
    ss_res_kde=np.sum((counts-y_pred_kde)**2)
    ss_tot=np.sum((counts - np.mean(counts))**2)
    r2_fit=1-(ss_res_fit/ss_tot)
    r2_kde=1-(ss_res_kde/ss_tot)

    ax.plot(r_grid,kde_vals,color='green',lw=2,label=f'KDE (R\u00b2={r2_kde:.4f})')
    ax.plot(r_grid,fit_vals,color='red',lw=2,label=f'Fitted PDF (\u03c3={sigma_fit:.2f}\u00b1{sigma_err:.2f}, R\u00b2={r2_fit:.4f})')
    ax.plot(r_grid,theory_vals,'k--',lw=1.5,label=f'Theory (\u03c3={std_dev_theory:.2f})')
    ax.set_title(f'{dim_label} final distance distribution (t={t_total})')
    ax.set_xlabel('Distance from origin (r)')
    ax.set_ylabel('Probability density')
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    print(f"\n{dim_label} distribution and fit analysis")
    print(f"Theoretical standard deviation: {std_dev_theory:.4f} | Fitted standard deviation: {sigma_fit:.4f}\u00b1{sigma_err:.4f}")
    print(f"Goodness of fit -> KDE R\u00b2: {r2_kde:.5f} | Parametric R\u00b2: {r2_fit:.5f}")
    print(f"Numerical integration check (total probability) -> KDE: {area_kde:.4f}, Fit: {area_fit:.4f}")
plt.show()