import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import  time

time_start = time.time()


# define variables
N_generator = 1 # the number of generators
N_nongenerator = 2 # the number of non generators
N = N_generator + N_nongenerator
#A_0 = np.load("adjacent_matrix_100_edges.npy")
A_0 = np.array([[0,1,1],[1,0,0],[1,0,0]])   # for motif 1
#A_0 = np.array([[0,1,1],[1,0,1],[1,1,0]])  # for motif 2
#A_0 = np.array([[0,1,0],[1,0,1],[0,1,0]])  # for motif 3

K = 1.0
M_0 = 1.0
M_array = np.ones((N,1))*M_0  # define M matrix
P_matrix = np.zeros((N,1))
P_matrix[0:N_generator,[0]] = -np.ones((N_generator,1))*0.9/float(N_generator)
P_matrix[N_generator:N,[0]] = np.ones((N_nongenerator,1))*0.9/float(N_nongenerator)

def fun_omega(K_temp,M_array, P_temp, A_temp,N, omega_temp, theta_temp):
    k_temp_omega = -omega_temp / M_array+ P_temp / M_array-K_temp*(np.sum(np.multiply(A_temp,
                   np.sin(theta_temp - (theta_temp).T)), axis=1,keepdims=True)).reshape(N,1) / M_array
    return k_temp_omega


def fun_theta(omega_temp):
    k_temp_theta = omega_temp
    return k_temp_theta

def get_truncated_normal(mean=0.0,sd=1.0,low=-1.0,upp=1.0):  #get a normal distribution within a range
    return truncnorm(
        (low-mean)/sd,(upp-mean)/sd,loc=mean,scale=sd)

Total_steps = 10000
h_step = 0.01


np.random.seed(1)
X = get_truncated_normal(mean=0.0, sd=1.0, low=-0.1, upp=0.1)
omega_matrix = X.rvs((N, 1))
#omega_matrix = np.zeros((N,1))
theta_matrix = np.zeros((N,1))


for i in range(0, Total_steps):
    k1_omega = h_step*fun_omega(K,M_array, P_matrix, A_0,N,omega_matrix, theta_matrix)
    k1_theta = h_step*fun_theta(omega_matrix)

    k2_omega = h_step*fun_omega(K,M_array, P_matrix, A_0,N,omega_matrix + k1_omega/ 2.0,theta_matrix + k1_theta/ 2.0,)
    k2_theta = h_step*fun_theta(omega_matrix + k1_omega / 2.0)

    k3_omega = h_step*fun_omega(K,M_array, P_matrix, A_0,N,omega_matrix + k2_omega / 2.0,theta_matrix + k2_theta / 2.0)
    k3_theta = h_step*fun_theta(omega_matrix + k2_omega / 2.0)

    k4_omega = h_step*fun_omega(K,M_array, P_matrix, A_0,N,omega_matrix + k3_omega,theta_matrix + k3_theta)
    k4_theta = h_step*fun_theta(omega_matrix+ k3_omega)

    omega_matrix = omega_matrix + (k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega) / 6.0
    theta_matrix = theta_matrix + (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta) / 6.0

    if np.max(np.abs(omega_matrix))<10**(-12):
        print(np.max(np.abs(omega_matrix)))
        break

theta_stable = theta_matrix

cm = plt.cm.get_cmap('RdYlBu_r')
#cm = plt.cm.get_cmap('Greens')

num_points = 200
sigma = 0.5
mu = 0.2
np.random.seed(10)
omega_perturb = np.random.normal(mu,sigma,num_points)
np.random.seed(20)
theta_perturb = np.random.normal(mu,sigma,num_points)+theta_stable[0,0]
H_numerical = np.load("E_expen_H_motif_1_random_points_nonlinear_simulation.npy")

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")

sc = plt.scatter(omega_perturb, theta_perturb, c=H_numerical[0,:], vmin=0, vmax=np.max(H_numerical[0,:]), s=35, cmap=cm)
plt.xlabel(r'$\omega_1$',fontsize=14)
plt.ylabel(r'$\theta_1$',fontsize=14)
plt.colorbar(sc)
plt.show()
