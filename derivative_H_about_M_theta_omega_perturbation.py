import numpy as np
import time
from scipy.stats import truncnorm
import numpy.linalg as linalg
from scipy.linalg import eig
import networkx as nx
import math
import scipy.io
import matplotlib.pyplot as plt

time_start = time.time()





# define variables
N_generator = 30 # the number of generators
N_nongenerator = 70 # the number of non generators
N = N_generator + N_nongenerator



#ER = nx.random_graphs.erdos_renyi_graph(N, 0.05)
#A_0 = nx.to_numpy_matrix(ER)
#np.save("adjacent_matrix_100_edges.npy",A_0)
A_0 = np.load("adjacent_matrix_100_edges.npy")
#scipy.io.savemat('adjacent_matrix_100_edges.mat',{'A':A_0})

def get_truncated_normal(mean=0.0,sd=1.0,low=-1.0,upp=1.0):  #get a normal distribution within a range
    return truncnorm(
        (low-mean)/sd,(upp-mean)/sd,loc=mean,scale=sd)

np.random.seed(1)
X = get_truncated_normal(mean=0.0, sd=1.0, low=0.0, upp=0.05)
delta_P0 = X.rvs((N, 1))
V_matrix = np.diag(delta_P0[:, 0])

np.random.seed(10)
X = get_truncated_normal(mean=0.0, sd=1.0, low=0.0, upp=0.05)
delta_P1 = X.rvs((N, 1))
V_matrix_1 = np.diag(delta_P0[:, 0])


K = 1.0
h_step = 0.01
Total_steps = 8000
num_points = 8
H_derivative = np.zeros(num_points)
M_range = np.linspace(0.5,10.0,num_points)
#M_range = np.array([0.5,2.0,10.0])

for  k in range(0,num_points):
    M_0 = M_range[k]
    M_matrix = np.eye(N)*M_0  # define M matrix
    S_marix = M_matrix
    D_matrix = np.eye(N)*1.0

    L_matrix = np.diag(np.sum(A_0,axis=0))*1.0-A_0*1.0




    A_matrix = np.zeros((N+N,N+N))
    A_matrix[:N,N:] = np.eye(N)*1.0
    A_matrix[N:,:N] = -K*np.dot(linalg.inv(M_matrix),L_matrix)
    A_matrix[N:,N:] = -np.dot(linalg.inv(M_matrix),D_matrix)
    B_matrix = np.zeros((N+N,N))
    B_matrix[:N,:] = V_matrix_1**0.5
    B_matrix[N:,:] = np.dot(linalg.inv(M_matrix),V_matrix**0.5)

    [eigenValues_A,eigenVectors_AL,eigenVectors_AR] = eig(A_matrix,right=True,left=True,overwrite_a=True,overwrite_b=True,check_finite=True)
    idx = eigenValues_A.argsort()[::-1]
    eigenValues_A = eigenValues_A[idx]
    eigenVectors_AL = eigenVectors_AL[:,idx]
    eigenVectors_AR = eigenVectors_AR[:,idx]
    A_temp = np.dot(np.conj(eigenVectors_AL).T,eigenVectors_AR)
    eigenVectors_AR = np.dot(eigenVectors_AR,np.linalg.inv(np.diag(np.diag(A_temp))))

    C_matrix = np.zeros((N+N,N+N))
    C_matrix[:N,:N] = 2.0*L_matrix
    C_matrix[N:,N:] = 0.5*S_marix

    u_i = np.zeros((N,1))
    u_i[0] = 1.0
    Phi_matrix = np.diag(u_i[:,0])
    A_derivative = np.zeros((N+N,N+N))
    A_derivative[N:,:N] = K*np.dot(Phi_matrix,np.dot((linalg.inv(M_matrix))**2,L_matrix))
    A_derivative[N:,N:] = np.dot(Phi_matrix,np.dot((linalg.inv(M_matrix))**2,D_matrix))
    B_derivative = np.zeros((N+N,N))
    B_derivative[N:,:] = -np.dot(Phi_matrix,np.dot((linalg.inv(M_matrix))**2,V_matrix**0.5))
    C_derivative = np.zeros((N+N,N+N))
    C_derivative[N:, N:] = 0.5 * Phi_matrix

    eta_matrix = np.eye(N)*1.0
    H_temp = np.zeros((N,1))
    #H_temp =0.0+0.0j
    for m in range(0,1):
        basis_m = eta_matrix[:,[m]]
        P_matrix = np.zeros((N+N,N+N))
        for i in range(1,N+N):
            for j in range(1,N+N):
                P_temp =  np.dot(eigenVectors_AL[:,[i]], np.dot(np.conj(eigenVectors_AR[:, [i]].T),
                                np.dot(C_matrix,np.dot(eigenVectors_AR[:,[j]],np.conj(eigenVectors_AL[:,[j]].T)))))
                P_matrix = P_matrix - ((1.0/(eigenValues_A[i].conjugate() + eigenValues_A[j]))*P_temp).real

        P_derivative = np.zeros((N+N,N+N))
        Q_matrix = np.dot(P_matrix,A_derivative)+np.dot(A_derivative.T,P_matrix)+C_derivative
        for i in range(1,N+N):
            for j in range(1,N+N):
                P_derivative_temp =  np.dot(eigenVectors_AL[:,[i]], np.dot(np.conj(eigenVectors_AR[:, [i]].T),
                                np.dot(Q_matrix,np.dot(eigenVectors_AR[:,[j]],np.conj(eigenVectors_AL[:,[j]].T)))))
                P_derivative = P_derivative - ((1.0/(eigenValues_A[i].conjugate() + eigenValues_A[j]))*P_derivative_temp).real
        H_derivative_temp = np.dot(B_derivative.T,np.dot(P_matrix,B_matrix))+np.dot(B_matrix.T,np.dot(P_derivative,B_matrix))+np.dot(B_matrix.T,np.dot(P_matrix,B_derivative))
    H_derivative[k] = np.dot(basis_m.T,np.dot(H_derivative_temp,basis_m))
    print k

print H_derivative
np.save("Energy_expended_H_derivative_M_omega_theta_perturbation_S_is_M.npy",H_derivative)
#np.save("Energy_expended_H_derivative_M_omega_theta_perturbation.npy",H_derivative)
time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")

plt.plot(M_range,H_derivative,'*-',color='blue',label='theta_omega_perturbation',alpha=0.5)
plt.legend(fontsize=15)
plt.xlabel('$M$',fontsize=15)
plt.ylabel('$H$',fontsize=15)
plt.show()