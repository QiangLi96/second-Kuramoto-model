import numpy as np
import time
import matplotlib.pyplot as plt
import math
from scipy.stats import truncnorm
import numpy.linalg as linalg

time_start = time.time()

# define variables
N_generator = 1 # the number of generators
N_nongenerator = 2 # the number of non generators
N = N_generator + N_nongenerator
#A_0 = np.load("adjacent_matrix_100_edges.npy")
#A_0 = np.array([[0,1,1],[1,0,0],[1,0,0]])   # for motif 1
A_0 = np.array([[0,1,1],[1,0,1],[1,1,0]])  # for motif 2
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
delta_theta = np.zeros((N, 1))
delta_omega = np.zeros((N,1))
eta_matrix = np.eye(N)*1.0


num_points = 10
np.random.seed(10)
omega_perturb = np.linspace(-0.5,0.5,num_points)
np.random.seed(20)
theta_perturb = np.linspace(-0.5,0.5,num_points)



M_matrix = np.eye(N) * M_0  # define M matrix
S_matrix = M_matrix
B_matrix = np.zeros((N + N, N))
H_numerical_range = np.zeros((N,num_points,num_points))


for k_omg in range(0,num_points):
    for k in range(0,num_points):
        delta_P0 = np.ones((N, 1)) * omega_perturb[k_omg]
        V_matrix_2 = np.diag(delta_P0[:, 0])

        delta_P1 = np.ones((N, 1)) * theta_perturb[k]
        V_matrix_1 = np.diag(delta_P1[:, 0])
        B_matrix[:N, :] = V_matrix_1
        B_matrix[N:, :] = np.dot(linalg.inv(M_matrix), V_matrix_2)
        for j in range(0, 1):
            delta_theta = np.dot(B_matrix[:N,:], eta_matrix[:, [j]])
            delta_omega = np.dot(B_matrix[N:,:], eta_matrix[:, [j]])
            Sum = np.zeros((N, Total_steps + 1))

            theta_matrix = theta_stable+delta_theta
            omega_matrix = delta_omega
            Matrix_temp = np.diag(np.sum(A_0, axis=1)) - 2.0 * A_0
            Sum[:, [0]] = np.dot(np.diag(delta_theta[:,0]), np.dot(Matrix_temp, delta_theta[:, [0]])) + np.dot(A_0, np.dot(np.diag(delta_theta[:, 0]), delta_theta[:, [0]])) + \
                          0.5 * np.dot(np.diag(delta_omega[:, 0]), np.dot(S_matrix, delta_omega[:, [0]]))

            for i in range(0, Total_steps):
                k1_omega = h_step * fun_omega(K, M_array, P_matrix, A_0, N, omega_matrix, theta_matrix)
                k1_theta = h_step * fun_theta(omega_matrix)

                k2_omega = h_step * fun_omega(K, M_array, P_matrix, A_0, N, omega_matrix + k1_omega / 2.0,
                                              theta_matrix + k1_theta / 2.0, )
                k2_theta = h_step * fun_theta(omega_matrix + k1_omega / 2.0)

                k3_omega = h_step * fun_omega(K, M_array, P_matrix, A_0, N, omega_matrix + k2_omega / 2.0,
                                              theta_matrix + k2_theta / 2.0)
                k3_theta = h_step * fun_theta(omega_matrix + k2_omega / 2.0)

                k4_omega = h_step * fun_omega(K, M_array, P_matrix, A_0, N, omega_matrix + k3_omega,
                                              theta_matrix + k3_theta)
                k4_theta = h_step * fun_theta(omega_matrix + k3_omega)

                omega_matrix = omega_matrix + (k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega) / 6.0
                theta_matrix = theta_matrix + (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta) / 6.0
                #print np.max(np.abs(omega_matrix))
                delta_theta_temp = theta_matrix - theta_stable
                Sum[:, [i+1]] = np.dot(np.diag(delta_theta_temp[:,0]), np.dot(Matrix_temp, delta_theta_temp[:, [0]])) + np.dot(A_0, np.dot(np.diag(delta_theta_temp[:, 0]), \
                                delta_theta_temp[:, [0]])) + 0.5 * np.dot(np.diag(omega_matrix[:, 0]), np.dot(S_matrix, omega_matrix[:, [0]]))

            H_numerical_range[:,k,k_omg] = (h_step/3.0)*(Sum[:,0]+4.0*np.sum(Sum[:,1:Total_steps:2],axis=1)+2.0*np.sum(Sum[:,2:Total_steps-1:2],axis=1)+Sum[:,-1])
        print(k)
    print(k_omg)

np.save("E_ex_H_motif_2_random_imshow_nonlinear_simulation.npy",H_numerical_range)
#np.save("Energy_expen_H_motif_2_single_point_numerical_perturbation_range_nonlinear.npy",H_numerical_range)

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")