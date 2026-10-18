import numpy as np
import matplotlib.pyplot as plt  #This is used for plotting the training loss and the realized signal prediction error.
#First we use the setup given by the 
rng = np.random.default_rng(31019)
n, d = 20, 40
eigenvalues = np.geomspace(1.0, 1e-3, n)
V, _ = np.linalg.qr(rng.standard_normal((d, n)), mode="reduced")
X = np.sqrt(n) * (np.sqrt(eigenvalues)[:, None] * V.T)
signal = np.zeros(n)
signal[:4] = 3.0
sigma, eta = 1.0, 1.0
steps = np.array([0, 1, 2, 3, 5, 10, 20, 50, 100, 300, 1000, 3000])

# Part e
# Task: Generate one observation vector y; determine standard deviation and size according to the given parameters.
noise = rng.normal(loc=0.0, scale=sigma, size=n)
y = signal + noise #We generate y outside the loop to ensure that it remains consistent across all iterations of gradient descent.

# Optional verification: K = XX^T / n has the prescribed eigenvalues.
K = (X @ X.T) / n
print("Does K equal diag(eigenvalues)?", np.allclose(K, np.diag(eigenvalues)))

# Implementing Gradient Descent
w = np.zeros(d) # We initialize the weight vector w to zeros, a 40 dimensional vector, as starting point of the GD algorithm.
max_steps = int(np.max(steps)) 
iterations = np.arange(max_steps + 1) 
training_losses = np.zeros(max_steps + 1)
prediction_errors = np.zeros(max_steps + 1)
checkpoints = set(steps.tolist())
discrepancies = {}

for k in iterations:
    # Prediction using the CURRENT iterate w_k. The rest is just going with the formula. 
    training_losses[k] = np.linalg.norm(X @ w - y) ** 2 / (2 * n)
    prediction_errors[k] = np.linalg.norm(X @ w - signal) ** 2 / n
    if k in checkpoints: #These checkpoints are used for us to compare GD with the spectral formula.
        # Since U=I, the spectral formula works coordinate-by-coordinate.
        spectral_prediction = (1.0 - (1.0 - eta * eigenvalues) ** k) * y
        discrepancies[int(k)] = np.linalg.norm(X @ w - spectral_prediction)
    # Stop after recording k = 3000.
    if k == max_steps:
        break
    # Gradient of the empirical square loss with respect to w.
    gradient = X.T @ (X @ w - y) / n
    w = w - eta * gradient

print("\nCheckpoint comparison")
print(f"{'k':>6} {'||GD - spectral||_2':>23} {'training loss':>17} {'prediction error':>19}")
for k in steps:
    k = int(k)
    print(f"{k:6d} {discrepancies[k]:23.6e} "
          f"{training_losses[k]:17.8f} {prediction_errors[k]:19.8f}")

max_discrepancy = max(discrepancies.values())
best_k = int(np.argmin(prediction_errors))
print(f"\nLargest Euclidean discrepancy: {max_discrepancy:.6e}")
print(f"Smallest realized signal error: {prediction_errors[best_k]:.8f} (at k={best_k})")
print(f"Final training loss: {training_losses[-1]:.8f}")
print(f"Final realized signal error: {prediction_errors[-1]:.8f}")

plt.figure(figsize=(7, 4.5))
plt.plot(iterations, training_losses, linewidth=1.8)
plt.xscale("symlog", linthresh=1)
plt.xlim(0, max_steps)
plt.xticks([0, 1, 3, 10, 100, 1000, 3000],
           ["0", "1", "3", "10", "100", "1000", "3000"])
plt.xlabel("GD iteration k")
plt.ylabel(r"Training loss  $\|Xw_k-y\|_2^2/(2n)$")
plt.title("Gradient descent: training loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gd_training_loss.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(iterations, prediction_errors, linewidth=1.8)
plt.axvline(best_k, linestyle="--", alpha=0.7,
            label=f"Minimum at k={best_k}")
plt.xscale("symlog", linthresh=1)
plt.xlim(0, max_steps)
plt.xticks([0, 1, 3, 10, 100, 1000, 3000],
           ["0", "1", "3", "10", "100", "1000", "3000"])
plt.xlabel("GD iteration k")
plt.ylabel(r"Realized signal error  $\|Xw_k-s\|_2^2/n$")
plt.title("Gradient descent: realized signal prediction error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gd_signal_error.png", dpi=180)
plt.close()

# Part f Generate 500 independent noise vectors
num_datasets = 500
noises = rng.normal(
    loc=0.0,
    scale=sigma,
    size=(num_datasets, n)
)
# This gives us a matrix of shape (500,20) where each row is a different noise vector. 
observations = signal[None, :] + noises
all_k = np.arange(int(steps.max()) + 1)

# g_k(lambda_i) = 1 - (1 - eta*lambda_i)^k
filters = 1.0 - (
    1.0 - eta * eigenvalues[None, :]
) ** all_k[:, None]

# Squared bias:
# B_k^2 = (1/n) sum_i [(1-g_k(lambda_i))*s_i]^2
squared_bias = np.sum(
    ((1.0 - filters) * signal[None, :]) ** 2,
    axis=1
) / n

# Variance:
# V_k = (sigma^2/n) sum_i [g_k(lambda_i)]^2
variance = sigma ** 2 * np.sum(
    filters ** 2, axis=1
) / n

# Total theoretical risk
theoretical_risk = squared_bias + variance

# The risk of exact interpolation
interpolation_risk = sigma ** 2

empirical_risk = np.empty(len(steps))
mc_standard_error = np.empty(len(steps))

for j, k in enumerate(steps):
    # Spectral weights at iteration k
    spectral_weights = filters[k]
    # Each row is the prediction for one dataset.
    predictions = observations * spectral_weights
    # Calculate 500 separate signal prediction errors.
    errors = np.sum(
        (predictions - signal) ** 2,
        axis=1
    ) / n
    # Average the 500 errors to estimate expected risk.
    empirical_risk[j] = np.mean(errors)
    # Optional: Monte Carlo standard error of the mean.
    mc_standard_error[j] = (
        np.std(errors, ddof=1) / np.sqrt(num_datasets)
    )

print("\n k     empirical        theory     bias^2     variance      MC SE")

for j, k in enumerate(steps):
    print(
        f"{k:4d}  {empirical_risk[j]:12.6f}"
        f"  {theoretical_risk[k]:12.6f}"
        f"  {squared_bias[k]:9.6f}"
        f"  {variance[k]:11.6f}"
        f"  {mc_standard_error[j]:9.6f}"
    )

best_listed_index = int(
    np.argmin(theoretical_risk[steps])
)
best_listed_k = int(steps[best_listed_index])

print(f"\nInterpolation risk = {interpolation_risk:.6f}")

print(
    f"Best theoretical risk among the listed k: "
    f"R_{best_listed_k} = "
    f"{theoretical_risk[best_listed_k]:.6f}"
)

print(
    f"Empirical risk at that k: "
    f"{empirical_risk[best_listed_index]:.6f}"
)

print(
    "Is finite-iteration risk below interpolation risk?",
    theoretical_risk[best_listed_k] < interpolation_risk
)

print(
    "Largest absolute MC versus theory deviation =",
    np.max(np.abs(
        empirical_risk - theoretical_risk[steps]
    ))
)

#For the first 20 iterations.
plt.figure(figsize=(10, 6))

plt.plot(
    all_k, theoretical_risk,
    linewidth=2.2,
    label="Exact theoretical risk"
)

plt.plot(
    all_k, squared_bias,
    linestyle="--", linewidth=1.8,
    label="Theoretical squared bias"
)

plt.plot(
    all_k, variance,
    linestyle="-.", linewidth=1.8,
    label="Theoretical variance"
)

plt.axhline(
    interpolation_risk,
    linestyle=":", linewidth=2,
    label=f"Exact interpolation risk = {interpolation_risk:g}"
)

plt.scatter(
    steps, empirical_risk,
    s=54, marker="o", zorder=5,
    color="tab:purple",
    edgecolors="white", linewidths=0.5,
    label="Empirical risk (500 observations)"
)

# Symmetric log allows k=0 and gives early k more space.
plt.xscale("symlog", linthresh=5)
plt.xlim(0, int(steps.max()))

tick_values = [0, 1, 2, 3, 5, 10, 20,
               50, 100, 300, 1000, 3000]

plt.xticks(
    tick_values,
    [str(t) for t in tick_values],
    fontsize=9, rotation=40
)

plt.xlabel("Iteration k (symmetric-log scale)")
plt.ylabel(
    r"Risk $R_k = \mathbb{E}[\|Xw_k-s\|_2^2/n]$"
)
plt.title(
    "Gradient descent: empirical vs. theoretical prediction risk"
)
plt.grid(alpha=0.25)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("gd_part_f_risks.png", dpi=200)
plt.show()

plt.figure(figsize=(8.5, 5))

plt.plot(
    all_k[:21], theoretical_risk[:21],
    linewidth=2.2, label="Theoretical total risk"
)
plt.plot(
    all_k[:21], squared_bias[:21],
    linestyle="--", label="Squared bias"
)
plt.plot(
    all_k[:21], variance[:21],
    linestyle="-.", label="Variance"
)
plt.axhline(
    interpolation_risk,
    linestyle=":", label="Exact interpolation risk"
)

mask = steps <= 20

plt.scatter(
    steps[mask], empirical_risk[mask],
    s=55, zorder=5,
    color="tab:purple",
    edgecolors="white", linewidths=0.5,
    label="Empirical risk (500 observations)"
)

plt.xlim(0, 20)
plt.xticks(np.arange(0, 21, 2))
plt.xlabel("Iteration k")
plt.ylabel("Prediction risk")
plt.title("Early-iteration close-up")
plt.grid(alpha=0.25)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("gd_part_f_early_zoom.png", dpi=200)
plt.show()
