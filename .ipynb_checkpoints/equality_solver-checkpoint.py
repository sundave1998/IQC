import numpy as np
import cvxpy as cp
import tqdm as tqdm
import matplotlib.pyplot as plt

# # Test using sample problem
# u_f = 5
# L_f = 10

# u_phi = 1
# L_phi = 1

# lambda_2 = 0


# # M_f = np.zeros((7,7))
# # M_f[4,4]= -(u_f*L_f)/(u_f+L_f)
# # M_f[4,5]= 1/2
# # M_f[5,4]= 1/2
# # M_f[5,5]= -(1)/(u_f+L_f)


# # M_phi = np.zeros((7,7))
# # M_phi[4,4]= -(u_phi*L_phi)/(u_phi+L_phi)
# # M_phi[4,1]= 1/2
# # M_phi[1,4]= 1/2
# # M_phi[1,1]= -(1)/(u_phi+L_phi)


# # M_lambda = np.zeros((7,7))
# # M_lambda[3,3]= -1
# # M_lambda[1,1]= lambda_2
# # # eta = 0.2
# M_f = np.zeros((5,5))
# M_f[2,2]= -(u_f*L_f)/(u_f+L_f)
# M_f[2,3]= 1/2
# M_f[3,2]= 1/2
# M_f[3,3]= -(1)/(u_f+L_f)


# M_phi = np.zeros((5,5))
# M_phi[2,2]= -(u_phi*L_phi)/(u_phi+L_phi)
# M_phi[2,0]= 1/2
# M_phi[0,2]= 1/2
# M_phi[0,0]= -(1)/(u_phi+L_phi)


# M_lambda = np.zeros((5,5))
# M_lambda[4,4]= -1
# M_lambda[0,0]= lambda_2
# # print(M_0)
# # print(M_f)
# # print(M_phi)
# # print(M_lambda)



# A_1 = np.zeros((2,2))
# A_1[1,0] = A_1[1,1] = 1
# A_1_eta_part = np.zeros((2,2))
# A_1_eta_part[0,1] = -1
# A_2 = np.zeros((2,2))
# A_2[0,0] = A_2[1,1] = 1
# A_2_eta_part = np.zeros((2,2))
# A_1_eta_part[0,1] = -1

# B_1 = np.zeros((2,3))
# B_1[0,2] = 1
# B_1[1,2] = -1
# B_1_eta_part = np.zeros((2,3))
# B_1_eta_part[0,1] = -1
# B_2 = np.zeros((2,3))
# B_2[0,2] = 1
# B_2[1,2] = -1
# B_2_eta_part = np.zeros((2,3))
# B_2_eta_part[0,1] = -1


# FG_2 = np.zeros((2, 5))
# FG_2[0,1] = FG_2[1,4] = 1



# R_2 = np.zeros((5,3))
# R_2[0,0] = R_2[2,1] = R_2[3,2] = 1

# sigma_f = cp.Variable()
# sigma_phi = cp.Variable()
# sigma_lambda = cp.Variable()
# # sigma_FG = cp.Variable()
# # rho = cp.Variable()
# rho = 0.999
# # eta = cp.Variable()
# eta = 0.1
# P = cp.Variable((2, 2))
# P_constraint = cp.Variable((2, 2))
# # P = np.eye(2)
# MAT_temp_1 = cp.Variable((5, 5))
# MAT_temp_2 = cp.Variable((5, 5))


# # constraints += [rho >= 0]
# # constraints += [eta >= 0]
# # constraints += [rho <= 1]
# constraints = [sigma_f >= 0]
# constraints += [sigma_phi >= 0]
# constraints += [sigma_lambda >= 0]
# constraints += [P >> 0]
# # questionable, P has to be PD
# constraints += [P[0,0] == 1]
# # otherwise it's homogeneous and has multiple optimal solutions
# constraints += [(-MAT_temp_1 - sigma_f*M_f - sigma_phi*M_phi - sigma_lambda*M_lambda) >> 0]
# constraints += [MAT_temp_1[0:2,0:2] == ((A_1 + eta*A_1_eta_part).T@P@(A_1 + eta*A_1_eta_part) - rho*P)]
# constraints += [MAT_temp_1[0:2,2:5] == ((A_1 + eta*A_1_eta_part).T@P@(B_1 + eta*B_1_eta_part))]
# constraints += [MAT_temp_1[2:,:2] == ((B_1 + eta*B_1_eta_part).T@P@(A_1 + eta*A_1_eta_part))]
# constraints += [MAT_temp_1[2:,2:] == ((B_1 + eta*B_1_eta_part).T@P@(B_1 + eta*B_1_eta_part))]

# constraints += [(-MAT_temp_2 - sigma_f*M_f - sigma_phi*M_phi - sigma_lambda*M_lambda - FG_2.T@P_constraint@FG_2) >> 0]
# constraints += [MAT_temp_2[0:2,0:2] == ((A_2 + eta*A_2_eta_part).T@P@(A_2 + eta*A_2_eta_part) - rho*P)]
# constraints += [MAT_temp_2[0:2,2:5] == ((A_2 + eta*A_2_eta_part).T@P@(B_2 + eta*B_2_eta_part))]
# constraints += [MAT_temp_2[2:,:2] == ((B_2 + eta*B_2_eta_part).T@P@(A_2 + eta*A_2_eta_part))]
# constraints += [MAT_temp_2[2:,2:] == ((B_2 + eta*B_2_eta_part).T@P@(B_2 + eta*B_2_eta_part))]

# # (A_1 + eta*A_1_eta_part)
# # (A_2 + eta*A_2_eta_part)
# # (B_1 + eta*B_1_eta_part)
# # (B_2 + eta*B_2_eta_part)
# prob=cp.Problem(cp.Minimize(rho), constraints)
# # prob=cp.Problem(cp.Maximize(rho), constraints)
# prob.solve()


# print("The optimal value is", prob.value)
# print("A solution is")
# print(sigma_f.value, sigma_phi.value)
# print("P matrix is:", P.value)
# print(prob.value == 0)


IQC_best_list = np.zeros(10)
resolution = 100
lambda_num = 10

IQC_rho_plot = np.zeros((lambda_num, resolution))
for lambeda_choice in range(lambda_num):
    u_f = 1
    L_f = 2

    u_phi = 1
    L_phi = 2

    lambda_2 = (lambeda_choice/lambda_num)**2
#     lambda_2 = 0.01

    M_f = np.zeros((5,5))
    M_f[2,2]= -(u_f*L_f)/(u_f+L_f)
    M_f[2,3]= 1/2
    M_f[3,2]= 1/2
    M_f[3,3]= -(1)/(u_f+L_f)

    M_phi = np.zeros((5,5))
    M_phi[2,2]= -(u_phi*L_phi)/(u_phi+L_phi)
    M_phi[2,0]= 1/2
    M_phi[0,2]= 1/2
    M_phi[0,0]= -(1)/(u_phi+L_phi)

    M_lambda = np.zeros((5,5))
    M_lambda[4,4]= -1
    M_lambda[0,0]= lambda_2

    A_1 = np.zeros((2,2))
    A_1[1,0] = A_1[1,1] = 1
    A_1_eta_part = np.zeros((2,2))
    A_1_eta_part[0,1] = -1
    A_2 = np.zeros((2,2))
    A_2[0,0] = A_2[1,1] = 1
    A_2_eta_part = np.zeros((2,2))
    A_1_eta_part[0,1] = -1

    B_1 = np.zeros((2,3))
    B_1[0,2] = 1
    B_1[1,2] = -1
    B_1_eta_part = np.zeros((2,3))
    B_1_eta_part[0,1] = -1
    B_2 = np.zeros((2,3))
    B_2[0,2] = 1
    B_2[1,2] = -1
    B_2_eta_part = np.zeros((2,3))
    B_2_eta_part[0,1] = -1
    
    FG_2 = np.zeros((2, 5))
    FG_2[0,1] = FG_2[1,4] = 1
    # R_2 = np.zeros((5,3))
    # R_2[0,0] = R_2[2,1] = R_2[3,2] = 1
    sigma_f = cp.Variable()
    sigma_phi = cp.Variable()
    sigma_lambda = cp.Variable()
    # sigma_FG = cp.Variable()
    # rho = cp.Variable()
    # rho = 0.999
    # eta = cp.Variable()
    # eta = 0.1
    P = cp.Variable((2, 2))
    P_constraint = cp.Variable((2, 2))
    # P = np.eye(2)
    MAT_temp_1 = cp.Variable((5, 5))
    MAT_temp_2 = cp.Variable((5, 5))
    
    eta_selection = np.linspace(0,1,resolution,endpoint=False)
    smallest_rho = 0.5
    rho_selection = np.linspace(smallest_rho,1,resolution,endpoint=False)
    viable = np.zeros((resolution,resolution))
    for i, eta in tqdm.tqdm(enumerate(eta_selection)):
        for j, rho in enumerate(rho_selection):
            constraints = [sigma_f >= 0]
            constraints += [sigma_phi >= 0]
            constraints += [sigma_lambda >= 0]
            constraints += [P >> 0]
            # questionable, P has to be PD
            constraints += [P[0,0] == 1]
            # otherwise it's homogeneous and has multiple optimal solutions
            constraints += [(-MAT_temp_1 - sigma_f*M_f - sigma_phi*M_phi - sigma_lambda*M_lambda) >> 0]
            constraints += [MAT_temp_1[0:2,0:2] == ((A_1 + eta*A_1_eta_part).T@P@(A_1 + eta*A_1_eta_part) - rho*P)]
            constraints += [MAT_temp_1[0:2,2:5] == ((A_1 + eta*A_1_eta_part).T@P@(B_1 + eta*B_1_eta_part))]
            constraints += [MAT_temp_1[2:,:2] == ((B_1 + eta*B_1_eta_part).T@P@(A_1 + eta*A_1_eta_part))]
            constraints += [MAT_temp_1[2:,2:] == ((B_1 + eta*B_1_eta_part).T@P@(B_1 + eta*B_1_eta_part))]

            constraints += [((-MAT_temp_2 - sigma_f*M_f - sigma_phi*M_phi - sigma_lambda*M_lambda- FG_2.T@P_constraint@FG_2)) >> 0]
            constraints += [MAT_temp_2[0:2,0:2] == ((A_2 + eta*A_2_eta_part).T@P@(A_2 + eta*A_2_eta_part) - rho*P)]
            constraints += [MAT_temp_2[0:2,2:5] == ((A_2 + eta*A_2_eta_part).T@P@(B_2 + eta*B_2_eta_part))]
            constraints += [MAT_temp_2[2:,:2] == ((B_2 + eta*B_2_eta_part).T@P@(A_2 + eta*A_2_eta_part))]
            constraints += [MAT_temp_2[2:,2:] == ((B_2 + eta*B_2_eta_part).T@P@(B_2 + eta*B_2_eta_part))]

            # (A_1 + eta*A_1_eta_part)
            # (A_2 + eta*A_2_eta_part)
            # (B_1 + eta*B_1_eta_part)
            # (B_2 + eta*B_2_eta_part)
            prob=cp.Problem(cp.Minimize(rho), constraints)
    #         prob.setSolverParam("numThreads", 4)
            # prob=cp.Problem(cp.Maximize(rho), constraints)
            try:
                prob.solve()
#                 prob.solve(solver=cp.MOSEK, mosek_params={mosek.iparam.num_threads: 8})
#                 prob.solve(solver=cp.MOSEK)
            except:
                continue
            if prob.status not in ["infeasible", "unbounded"]:
    #             print("Problem is solvable for eta, rho:", eta, rho)
                viable[i,j] = 1
    best_eta = eta_selection[np.argmax(np.sum(viable, axis=1))]
    # print(best_eta)
    best_rho = rho_selection[np.argmax(viable[np.argmax(np.sum(viable, axis=1))])]
    print(best_rho)
    IQC_best_list[lambeda_choice] = best_rho
    for step in range(resolution):
        possible = viable[step,:]
        IQC_rho_plot[lambeda_choice, step] = np.sum(viable[step,:])
    if best_eta == 0:
        print("no solution")
        IQC_best_list[lambeda_choice] = 1
    plt.plot(eta_selection, 1-((1-smallest_rho)* IQC_rho_plot.T)/(resolution))
#     plt.imshow(np.flip(np.transpose(viable), axis=0))
#     plt.show()