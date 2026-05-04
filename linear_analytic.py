# JAX script: GD on R_lin,L and alignment with top eigenvector
#Figure 4b

import matplotlib
matplotlib.use('Agg')
import os
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("Current working directory:", os.getcwd())
base_dir = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base_dir, "results")

os.makedirs(results, exist_ok=True)
print("Saving to:", results)

# ---------------------------
# Helpers
# ---------------------------
def normalize(v):
    return v / (jnp.linalg.norm(v) + 1e-8)

# ---------------------------
# Parameters
# ---------------------------
d = 5
L = 6
lam = 0.01
stepsize = 1e-4
steps = 5000
n_runs = 10
seed = 0

# ---------------------------
# Random Sigma (SPD)
# ---------------------------
key = random.PRNGKey(seed)
key, subkey = random.split(key)

A = random.normal(subkey, (d, d))
Sigma = A @ A.T + 0.1 * jnp.eye(d)

TrSigma = jnp.trace(Sigma)
S2 = Sigma @ Sigma

# ---------------------------
# Define a(mu), b(mu)
# ---------------------------
def a_of(mu):
    return mu @ (Sigma @ mu)

def b_of(mu):
    return mu @ (S2 @ mu)

# ---------------------------
# Risk function
# ---------------------------
def R_lin(mu):
    a = a_of(mu)
    b = b_of(mu)

    return (
        TrSigma
        - 2 * lam * TrSigma * a
        - 2 * lam * (L + 1) * b
        + lam**2 * (L + 2) * TrSigma * a**2
        + lam**2 * (L + 2) * (L + 3) * a * b
    )

# automatic gradient 
#grad_R = jit(grad(R_lin))
def grad_R(mu):
    a = a_of(mu)
    b = b_of(mu)

    term1 = -4 * lam * TrSigma * (Sigma @ mu)
    term2 = -4 * lam * (L + 1) * (S2 @ mu)

    term3 = 4 * lam**2 * (L + 2) * TrSigma * a * (Sigma @ mu)

    term4 = 4 * lam**2 * (L + 2) * (L + 3) * (
        b * (Sigma @ mu) + a * (S2 @ mu)
    )

    return term1 + term2 + term3 + term4

# ---------------------------
# GD
# ---------------------------
def run_gd(mu_init):
    mu = mu_init
    mu_norm_history = []

    for i in range(steps):
        g = grad_R(mu)
        mu = mu - stepsize * g

        mu_norm = normalize(mu)
        mu_norm_history.append(mu_norm)

    return mu, mu_norm_history

# ---------------------------
# Top eigenvector
# ---------------------------
eigvals, eigvecs = jnp.linalg.eigh(Sigma)
true_top = normalize(eigvecs[:, -1])

# ---------------------------
# Logs
# ---------------------------
logs = {"Run": [], "Iterations": [], "Absolute cosine similarity": []}

# ---------------------------
# Multiple runs
# ---------------------------
for run in range(n_runs):
    key, subkey = random.split(key)

    mu_init = 0.01*random.normal(subkey, (d,))
    #mu_init = normalize(mu_init)

    mu_star, mu_norm_history = run_gd(mu_init)

    mu_norm_array = jnp.stack(mu_norm_history)
    sim_history = jnp.abs(jnp.dot(mu_norm_array, true_top))

    # logging
    for i in range(steps):
        logs["Run"].append(run + 1)
        logs["Iterations"].append(i + 1)
        logs["Absolute cosine similarity"].append(float(sim_history[i]))
        if i%2000==0:
            print(sim_history[i])

    print(f"Finished run {run+1}")

# ---------------------------
# Dataframe
# ---------------------------
df = pd.DataFrame(logs)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6))
sns.set_context("notebook", font_scale=1.5)

ax = sns.lineplot(
    data=df,
    x="Iterations",
    y="Absolute cosine similarity",
    legend=False,
    linewidth=3.0,
    alpha=0.7
)

plt.axhline(1, color='red', linestyle='--')

ax.set_ylabel("Absolute cosine similarity")
ax.set_xlabel("Iterations")
plt.title(r'Convergence of $\frac{\mu}{\|\mu\|}$ towards $\pm u_1$')

plt.grid(True)
plt.tight_layout()

filename = "plot_linear_risk_convergence_analytic.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()