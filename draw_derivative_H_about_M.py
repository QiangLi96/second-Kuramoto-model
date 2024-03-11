import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

time_start = time.time()


num_points = 8
M_range = np.linspace(0.5,10.0,num_points)
H_derivative_omega = np.load("Energy_expended_H_derivative_M_omega_perturbation_S_is_M.npy")
H_derivative_theta = np.load("Energy_expended_H_derivative_M_theta_perturbation_S_is_M.npy")
H_derivative_two = np.load("Energy_expended_H_derivative_M_omega_theta_perturbation_S_is_M.npy")


time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")

current_palette = sns.color_palette("Set2")
sns.set_palette(current_palette)
plt.figure(figsize=(8.0,4.8))
plt.plot(M_range,H_derivative_omega,'-o',label=r'perturb $\omega$',alpha=0.8)
plt.plot(M_range,H_derivative_theta,'-o',label=r'perturb $\theta$',alpha=0.8)
plt.plot(M_range,H_derivative_two,'-o',label=r'perturb $\omega+\theta$',alpha=0.8)
#plt.plot(M_range,H_derivative_omega,'*-',label='perturb omega',alpha=0.5)
#plt.plot(M_range,H_derivative_theta,'o-',label='perturb theta',alpha=0.5)
#plt.plot(M_range,H_derivative_two,'.-',label='perturb omega_theta',alpha=0.5)
plt.legend(fontsize=14,loc='upper left')
plt.xlabel(r'$m$',fontsize=14)
plt.ylabel(r'$\nabla_m H_1$',fontsize=14)
#plt.xlim(-2.5, 2.5)
plt.ylim(-0.15, 0.25)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('F:\Latex_work\work_2_2019_5_22\Figures_2\Fig__derivative_H1_about_M_Energy_expended.pdf')
#plt.savefig('F:\graduation_thesis\Figures\Fig__H1_Energy_expended_perturbation_range_omega_theta.eps',format='eps',dpi=1000)
plt.savefig('F:\Latex_work\work_2_2019_5_22\Figures_2\Fig__derivative_H1_about_M_range_Energy_expended.png')
#plt.savefig('Fig__H1_Energy_expended_M_range_C_eye_theta_omega_perturbation.pdf')
plt.show()