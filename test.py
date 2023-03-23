import numpy as np
import matplotlib.pyplot as plt

def prandtl_meyer_function(M, gamma):
    nu = np.sqrt((gamma+1)/(gamma-1)) * np.arctan(np.sqrt((gamma-1)/(gamma+1)*(M**2-1))) - np.arctan(np.sqrt(M**2-1))
    return nu

def first_derivative_exact(M, gamma):
    nu_theta = np.sqrt((gamma+1)/(gamma-1)) * ((gamma-1)/(gamma+1)*(M**2-1)+1)**(-1/2)
    d_theta_d_M = np.sqrt((gamma-1)/(gamma+1)) * M / ((M**2-1)*(1+((gamma-1)/(gamma+1))*(M**2-1))**(-1/2))
    d_nu_d_M = 0.5 * np.sqrt((gamma+1)/(gamma-1)) * nu_theta * d_theta_d_M
    return d_nu_d_M

def second_derivative_exact(M, gamma):
    nu_theta = np.sqrt((gamma+1)/(gamma-1)) * ((gamma-1)/(gamma+1)*(M**2-1)+1)**(-1/2)
    d_theta_d_M = np.sqrt((gamma-1)/(gamma+1)) * M / ((M**2-1)*(1+((gamma-1)/(gamma+1))*(M**2-1))**(-1/2))
    d2_theta_d_M2 = np.sqrt((gamma-1)/(gamma+1)) * ((M**2-1)*(1+((gamma-1)/(gamma+1))*(M**2-1))**(-3/2)) * ((gamma-1)/(gamma+1) + 2*M**2*(gamma-1)/(gamma+1)*(M**2-1))
    d_nu_theta_d_M = 0.5 * np.sqrt((gamma+1)/(gamma-1)) * nu_theta * d_theta_d_M
    d2_nu_d_M2 = 0.5 * np.sqrt((gamma+1)/(gamma-1)) * (d_nu_theta_d_M * d_theta_d_M + nu_theta * d2_theta_d_M2)
    return d2_nu_d_M2

def central_difference(func, M, gamma, h):
    return (func(M+h, gamma) - func(M-h, gamma)) / (2*h)

def forward_difference_1(func, M, gamma, h):
    return (func(M+h, gamma) - func(M, gamma)) / h

def forward_difference_2(func, M, gamma, h):
    return (-3*func(M, gamma) + 4*func(M+h, gamma) - func(M+2*h, gamma)) / (2*h)

gamma = 1.4
M_list = np.linspace(1, 5, 20)
h_list = [10**(-i) for i in range(1, 17)]
d_nu_d_M_exact = [first_derivative_exact(M, gamma) for M in M_list]
d2_nu_d_M2_exact = [second_derivative_exact(M, gamma) for M in M_list]

# First derivative approximations
for method in [central_difference, forward_difference_1, forward_difference_2]:
    d_nu_d_M_approx = [method(prandtl_meyer_function, M, gamma, h) for M in M_list for h in h_list]
    error = np.abs(d_nu_d_M_exact - d_nu_d_M_approx)
error_grid = error.reshape((len(M_list), len(h_list)))
plt.figure()
for i, h in enumerate(h_list):
    plt.semilogy(M_list, error_grid[:, i], label=f"h={h}")
    plt.semilogy(M_list, np.abs(d_nu_d_M_exact - d_nu_d_M_approx[-len(M_list):]), 'k--', label="exact")
    plt.title(f"First derivative approximation error ({method.name} method)")
    plt.xlabel("Mach number")
    plt.ylabel("Absolute error")
    plt.legend()
# Second derivative approximations

for method in [central_difference, forward_difference_2]:
    d2_nu_d_M2_approx = [method(prandtl_meyer_function, M, gamma, h) for M in M_list for h in h_list]
    error = np.abs(d2_nu_d_M2_exact - d2_nu_d_M2_approx)
    error_grid = error.reshape((len(M_list), len(h_list)))
plt.figure()
for i, h in enumerate(h_list):
    plt.semilogy(M_list, error_grid[:, i], label=f"h={h}")
    plt.semilogy(M_list, np.abs(d2_nu_d_M2_exact - d2_nu_d_M2_approx[-len(M_list):]), 'k--', label="exact")
    plt.title(f"Second derivative approximation error ({method.name} method)")
    plt.xlabel("Mach number")
    plt.ylabel("Absolute error")
    plt.legend()
# Error vs. h for M=1.8

M = 1.8
h_list = [10**(-i) for i in range(1, 9)]
d_nu_d_M_exact = first_derivative_exact(M, gamma)
d2_nu_d_M2_exact = second_derivative_exact(M, gamma)
error_1 = [np.abs(d_nu_d_M_exact - forward_difference_1(prandtl_meyer_function, M, gamma, h)) for h in h_list]
error_2 = [np.abs(d_nu_d_M_exact - central_difference(prandtl_meyer_function, M, gamma, h)) for h in h_list]
error_3 = [np.abs(d_nu_d_M_exact - forward_difference_2(prandtl_meyer_function, M, gamma, h)) for h in h_list]
plt.figure()
plt.loglog(h_list, error_1, 'r', label="forward difference 1")
plt.loglog(h_list, error_2, 'g', label="central difference")
plt.loglog(h_list, error_3, 'b', label="forward difference 2")
plt.title("First derivative error vs. h")
plt.xlabel("h")
plt.ylabel("Absolute error")
plt.legend()
error_1 = [np.abs(d2_nu_d_M2_exact - forward_difference_2(prandtl_meyer_function, M, gamma, h)) for h in h_list]
error_2 = [np.abs(d2_nu_d_M2_exact - central_difference(prandtl_meyer_function, M, gamma, h)) for h in h_list]
plt.figure()
plt.loglog(h_list, error_1, 'r', label="forward difference 2")
plt.loglog(h_list, error_2, 'g', label="central difference")
plt.title("Second derivative error vs. h")
plt.xlabel("h")
plt.ylabel("Absolute error")
plt.legend()

plt.show()