import numpy as np
import time
from scipy.stats import truncnorm
import numpy.linalg as linalg
from scipy.linalg import eig
import math
import scipy.io
import matplotlib.pyplot as plt

time_start = time.time()





# define variables
N_generator = 1 # the number of generators
N_nongenerator = 2 # the number of non generators
N = N_generator + N_nongenerator


#A_0 = np.load("adjacent_matrix_100_edges.npy")
#A_0 = np.array([[0,1,1],[1,0,0],[1,0,0]])  # for motif 1
A_0 = np.array([[0,1,1],[1,0,1],[1,1,0]])  # for motif 2
#A_0 = np.array([[0,1,0],[1,0,1],[0,1,0]])  #for motif 3


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

Total_steps = 20000
h_step = 0.01


np.random.seed(1)
X = get_truncated_normal(mean=0.0, sd=1.0, low=-0.5, upp=0.5)
omega_matrix = X.rvs((N, 1))
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


Laplace_matrix = np.cos(np.multiply(A_0,theta_stable-theta_stable.T))
Laplace_matrix = np.diag(np.sum(Laplace_matrix,axis=1))-Laplace_matrix

M_matrix = np.eye(N) * M_0  # define M matrix
S_matrix = M_matrix
D_matrix = np.eye(N) * 1.0

h_step = 0.01
Total_steps = 10000


num_points = 15
sigma = 0.1
mu_range = np.linspace(0.0,1.0,num_points)
H_theoretical_range = np.zeros((N,20,num_points))
eye_2 = np.zeros((N,N+N))
eye_2[:,:N] = np.eye(N)
eye_2[:,N:] = np.eye(N)
Trans_matrix = np.zeros((N,N+N))
Trans_matrix[:,:N] = A_0


for  k_p in range(0,num_points):
    np.random.seed(2 * k_p)
    omega_perturb = np.random.normal(mu_range[k_p], sigma, 20)
    np.random.seed(3 * k_p)
    theta_perturb = np.random.normal(mu_range[k_p], sigma, 20)
    for k in range(0,20):
        delta_P0 = np.ones((N, 1)) * omega_perturb[k]
        V_matrix = np.diag(delta_P0[:, 0])
        delta_P1 = np.ones((N, 1)) * theta_perturb[k]
        V_matrix_1 = np.diag(delta_P1[:, 0])

        L_matrix = Laplace_matrix
        A_matrix = np.zeros((N+N,N+N))
        A_matrix[:N,N:] = np.eye(N)*1.0
        A_matrix[N:,:N] = -K*np.dot(linalg.inv(M_matrix),L_matrix)
        A_matrix[N:,N:] = -np.dot(linalg.inv(M_matrix),D_matrix)
        B_matrix = np.zeros((N+N,N))
        B_matrix[:N,:] = V_matrix_1
        B_matrix[N:,:] = np.dot(linalg.inv(M_matrix),V_matrix)

        [eigenValues_A,eigenVectors_AL,eigenVectors_AR] = eig(A_matrix,right=True,left=True,overwrite_a=True,overwrite_b=True,check_finite=True)
        idx = eigenValues_A.argsort()[::-1]
        eigenValues_A = eigenValues_A[idx]
        eigenVectors_AL = eigenVectors_AL[:,idx]
        eigenVectors_AR = eigenVectors_AR[:,idx]
        A_temp = np.dot(np.conj(eigenVectors_AL).T,eigenVectors_AR)
        eigenVectors_AR = np.dot(eigenVectors_AR,np.linalg.inv(np.diag(np.diag(A_temp))))
        a_matrix = np.dot(eigenVectors_AR,np.dot(np.diag(eigenValues_A),np.conj(eigenVectors_AL).T))

        C_matrix = np.zeros((N+N,N+N))
        C_matrix[:N,:N] = np.diag(np.sum(A_0,axis=1))-2.0*A_0
        C_matrix[N:,N:] = 0.5*S_matrix
        eta_matrix = np.eye(N)*1.0
        for m in range(0,1):
            H_temp = np.zeros((N, 1))
            basis_m = eta_matrix[:,[m]]
            for i in range(1,N+N):
                for j in range(1,N+N):
                    #print np.shape()
                    temp_0 = (1.0/(eigenValues_A[i]+ eigenValues_A[j]))*np.dot(np.diag((np.dot(eigenVectors_AR[:,[i]],np.dot(np.conj(eigenVectors_AL[:,[i]].T),np.dot(B_matrix,basis_m))))[:,0]),\
                                    np.dot(C_matrix,np.dot(eigenVectors_AR[:,[j]],np.dot(np.conj(eigenVectors_AL[:,[j]].T),np.dot(B_matrix,basis_m)))))
                    temp_1 = (1.0/(eigenValues_A[i]+ eigenValues_A[j]))*np.dot(np.diag((np.dot(eigenVectors_AR[:,[i]],np.dot(np.conj(eigenVectors_AL[:,[i]].T),np.dot(B_matrix,basis_m))))[:,0]), \
                                    (np.dot(eigenVectors_AR[:, [j]],np.dot(np.conj(eigenVectors_AL[:, [j]].T), np.dot(B_matrix, basis_m)))))
                    H_temp = H_temp - np.dot(eye_2,temp_0)-np.dot(Trans_matrix,temp_1)
            H_theoretical_range[:,k,k_p] = H_temp[:,0]

        #print(k)
    print(k_p)

#print H_theoretical
np.save("Energy_expen_H_motif_2_single_point_theoretical_perturbation_range.npy",H_theoretical_range)

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")