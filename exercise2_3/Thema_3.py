import numpy as np

# Samples for class ω1
samples_omega1 = np.array([
    [0.42, -0.087, 0.58],
    [-0.2, -3.3, -3.4],
    [1.3, -0.32, 1.7],
    [0.39, 0.71, 0.23],
    [-1.6, -5.3, -0.15],
    [-0.029, 0.89, -4.7],
    [-0.23, 1.9, 2.2],
    [0.27, -0.3, -0.87],
    [-1.9, 0.76, -2.1],
    [0.87, -1, -2.6]
])

def calculate_mle(samples):
    # Mean (μ)
    mu_hat = np.mean(samples, axis=0)

    # Variance (σ^2)
    # axis=0 because we want to calculate the variance of each column 
    # ddof=0 because we want to calculate the maximum likelihood estimation -- Delta Degrees of Freedom
    sigma_hat_sq = np.var(samples, axis=0, ddof=0)  
    return mu_hat, sigma_hat_sq

mu_hat, sigma_hat_sq = calculate_mle(samples_omega1)
print("Mean (μ̂):", mu_hat)
print("Variance (σ̂^2):", sigma_hat_sq)


###################Part 2###################
def calculate_2d_mle(samples):
    # Mean (μ)
    mu_hat = np.mean(samples, axis=0)

    # Covariance matrix (Σ)
    # rowvar=False because each column represents a variable
    # bias=True for maximum likelihood estimation
    sigma_hat = np.cov(samples, rowvar=False, bias=True)
    return mu_hat, sigma_hat

# Pair (x1, x2)
samples_omega1_x1_x2 = samples_omega1[:, [0, 1]]
mu_hat_x1_x2, sigma_hat_x1_x2 = calculate_2d_mle(samples_omega1_x1_x2)
print("Mean (μ̂) for (x1, x2):", mu_hat_x1_x2)
print("Covariance (Σ̂) for (x1, x2):", sigma_hat_x1_x2)

# Pair (x1, x3)
samples_omega1_x1_x3 = samples_omega1[:, [0, 2]]
mu_hat_x1_x3, sigma_hat_x1_x3 = calculate_2d_mle(samples_omega1_x1_x3)
print("Mean (μ̂) for (x1, x3):", mu_hat_x1_x3)
print("Covariance (Σ̂) for (x1, x3):", sigma_hat_x1_x3)

# Pair (x2, x3)
samples_omega1_x2_x3 = samples_omega1[:, [1, 2]]
mu_hat_x2_x3, sigma_hat_x2_x3 = calculate_2d_mle(samples_omega1_x2_x3)
print("Mean (μ̂) for (x2, x3):", mu_hat_x2_x3)
print("Covariance (Σ̂) for (x2, x3):", sigma_hat_x2_x3)

###################Part 3###################
mu_hat_3d, sigma_hat_3d = calculate_2d_mle(samples_omega1)
print("Mean (μ̂) for 3D:", mu_hat_3d)
print("Covariance (Σ̂) for 3D:", sigma_hat_3d)

###################Part 4###################
# Samples for class ω2
samples_omega2 = np.array([
    [-0.4, 0.58, 0.089],
    [-0.31, 0.27, -0.04],
    [0.38, 0.055, -0.035],
    [-0.15, 0.53, 0.011],
    [-0.35, 0.47, 0.034],
    [0.17, 0.69, 0.1],
    [-0.011, 0.55, -0.18],
    [-0.27, 0.61, 0.12],
    [-0.065, 0.49, 0.0012],
    [-0.12, 0.054, -0.063]
])

mu_hat_diag, sigma_hat_diag_sq = calculate_mle(samples_omega2)
print("Mean (μ̂) for diagonal Σ:", mu_hat_diag)
print("Variances (σ̂^2) for diagonal Σ:", sigma_hat_diag_sq)

###################Part 5###################
print("Comparison of Mean Vectors (μ̂):")
print("1D (x1, x2, x3):", mu_hat)
print("2D (x1, x2):", mu_hat_x1_x2)
print("2D (x1, x3):", mu_hat_x1_x3)
print("2D (x2, x3):", mu_hat_x2_x3)
print("3D:", mu_hat_3d)
print("Diagonal Σ:", mu_hat_diag)

print("\nComparison of Variances (σ̂^2):")
print("1D:", sigma_hat_sq)
print("Diagonal Σ:", sigma_hat_diag_sq)


