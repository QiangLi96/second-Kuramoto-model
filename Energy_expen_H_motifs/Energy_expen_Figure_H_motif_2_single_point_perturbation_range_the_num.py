import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

time_start = time.time()

# define variables
N_generator = 1 # the number of generators
N_nongenerator = 2 # the number of non generators
N = N_generator + N_nongenerator

num_points = 15
sigma = 0.1
mu_range = np.linspace(0.0,1.0,num_points)

"""
H_1 = np.load("Energy_expen_H_motif_1_single_point_numerical_perturbation_range_nonlinear.npy")
H_2 = np.load("Energy_expen_H_motif_1_single_point_theoretical_perturbation_range.npy")
H_3 = np.load("Energy_expen_H_motif_2_single_point_numerical_perturbation_range_nonlinear.npy")
H_4 = np.load("Energy_expen_H_motif_2_single_point_theoretical_perturbation_range.npy")

delta_H_simu = H_1[:,0,:]-H_3[:,0,:]
delta_H_anal = H_2[:,0,:]-H_4[:,0,:]
print delta_H_simu
print delta_H_anal
"""

H_simulation_range = np.load("Energy_expen_H_motif_2_single_point_numerical_perturbation_range_nonlinear.npy")
H_analytic_range = np.load("Energy_expen_H_motif_2_single_point_theoretical_perturbation_range.npy")
H_simulation_average = np.sum(H_simulation_range,axis=1)/20
H_simulation_std = np.std(H_simulation_range,axis=1)
H_analytic_average = np.sum(H_analytic_range,axis=1)/20.0
H_analytic_std = np.std(H_analytic_range,axis=1)

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")
simulation_color_bar = ['mistyrose','honeydew']
analytic_color_bar = ['powderblue','thistle']
current_palette = sns.color_palette("Set1")
sns.set_palette(current_palette)
#plt.figure(figsize=(10.0,8.0))
for i in range(0,2):
    plt.figure(i)
    plt.plot(mu_range,H_simulation_average[i,:],label='$H_{'+str(i+1)+'1}$(simulation)')
    plt.plot(mu_range,H_analytic_average[i,:],label='$H_{'+str(i+1)+'1}$(analytic)')
    plt.fill_between(mu_range, H_simulation_average[i,:]-H_simulation_std[i,:], H_simulation_average[i,:]+H_simulation_std[i,:],color=sns.xkcd_rgb['grapefruit'],alpha=0.2)
    plt.fill_between(mu_range, H_analytic_average[i,:]-H_analytic_std[i, :], H_analytic_average[i,:]+H_analytic_std[i, :],color=sns.xkcd_rgb['sky blue'],alpha=0.2)
    plt.legend(fontsize=14,loc='upper left')
    plt.xlabel('$S$',fontsize=14)
    plt.ylabel('$H$',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('F:\Latex_work\work_2_2019_5_22\Figures\Fig_H1_Energy_expen_motif_2_'+str(i)+'_perturbation_range_omega_theta.pdf')
plt.show()
"""
plt.figure(3)
num_points = 10
np.random.seed(10)
omega_perturb = np.linspace(-0.5,0.5,num_points)
np.random.seed(20)
theta_perturb = np.linspace(-0.5,0.5,num_points)
y_theta,x_omega= np.meshgrid(theta_perturb,omega_perturb)

H_simulation = np.load("E_ex_H_motif_2_random_imshow_nonlinear_simulation.npy")
extent = [np.min(x_omega),np.max(x_omega),np.min(y_theta),np.max(y_theta)]
plt.imshow(H_simulation[0,:,:],extent=extent,cmap='Reds',vmin=0,vmax=3.6,origin="lower")
#plt.imshow(H_simulation[0,:,:],extent=extent,cmap='Blues',vmin=0,vmax=np.max(H_simulation[0,:,:]),origin="lower")
plt.xlabel(r'$\omega_1$',fontsize=14)
plt.ylabel(r'$\theta_1$',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.colorbar()
plt.savefig('F:\Latex_work\work_2_2019_5_22\Figures\Fig_H1_Energy_expen_motif_2_c_perturbation_range_omega_theta.pdf')

plt.figure(4)
im_motif2 = plt.imread("F:\Latex_work\work_2_2019_5_22\Figures\motif_2_2.png")
plt.imshow(im_motif2)
plt.axis('off')
plt.savefig('F:\Latex_work\work_2_2019_5_22\Figures\Fig_H1_Energy_expen_motif_2_d_perturbation_range_omega_theta.pdf')
#plt.savefig('F:\Latex_work\work_2_2019_5_22\Figures\Fig_H1_Energy_expen_motif_2_d_perturbation_range_omega_theta.png')
plt.show()
#plot error bar

"""