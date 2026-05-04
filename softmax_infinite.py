# JAX script: gradient descent on R(mu) and alignment with top eigenvector 
#Figure 3b

import matplotlib
matplotlib.use('Agg')
import os
import jax
import jax.numpy as jnp
from jax import random
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
    return v / jnp.linalg.norm(v)

# ---------------------------
# Parameters
# ---------------------------
dim = 5
lambda_ = 0.01
stepsize = 1e-3
steps = 3000
seed = 0
n_runs = 10

logs = {"Run": [], "Iterations": [], "Absolute cosine similarity": []}

# ---------------------------
# Random Sigma (SPD)
# ---------------------------
key = random.PRNGKey(seed)
key, subkey = random.split(key)

A = random.normal(subkey, (dim, dim))
Sigma = A @ A.T + 0.1 * jnp.eye(dim)

TrSigma = jnp.trace(Sigma)

# ---------------------------
# Define a(mu), b(mu)
# ---------------------------
def a_of(mu, S=Sigma):
    return mu @ (S @ mu)

def b_of(mu, S=Sigma):
    S2 = S @ S
    return mu @ (S2 @ mu)

# ---------------------------
# Analytic gradient of R
# ---------------------------
def grad_R_analytic(mu, S=Sigma, lambda_=lambda_):
    S2 = S @ S
    a = a_of(mu, S)
    b = b_of(mu, S)

    grad = -4 * lambda_ * (S2 @ mu) \
           + 2 * lambda_**2 * (a * (S2 @ mu) + b * (S @ mu))

    return grad

# ---------------------------
# Gradient Descent
# ---------------------------
def run_gd(mu_init, steps, stepsize):
    mu = mu_init
    mu_norm_history = []

    for i in range(steps):
        g = grad_R_analytic(mu)
        mu = mu - stepsize * g

        mu_norm = normalize(mu)
        mu_norm_history.append(mu_norm)

    return mu, mu_norm_history

# ---------------------------
# Top eigenvector of Sigma
# ---------------------------
eigvals, eigvecs = jnp.linalg.eigh(Sigma)
true_top = eigvecs[:, -1]

# ---------------------------
# Multiple runs
# ---------------------------
for run in range(n_runs):
    key, subkey = random.split(key)

    mu_init = random.normal(subkey, (dim,))
    mu_init = normalize(mu_init)

    mu_star, mu_norm_history = run_gd(mu_init, steps, stepsize)

    mu_norm_array = jnp.stack(mu_norm_history)
    sim_history = jnp.abs(jnp.dot(mu_norm_array, true_top))

    # logging
    for i in range(steps):
        logs["Run"].append(run + 1)
        logs["Iterations"].append(i + 1)
        logs["Absolute cosine similarity"].append(float(sim_history[i]))

    print(f"Finished run {run+1}")

# ---------------------------
# Dataframe
# ---------------------------
df = pd.DataFrame(logs)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6),label=r' $|\langle \frac{\mu_k}{\Vert\mu_k\Vert}, u_1 \rangle|$')
sns.set_context("notebook", font_scale=1.5)

ax = sns.lineplot(
    data=df,
    x="Iterations",
    y="Absolute cosine similarity",
    legend=False,
    linewidth=2.0,
    alpha=0.7
)

# horizontal line at 1
plt.axhline(1, color='red', linestyle='--')

ax.set_ylabel("Absolute cosine similarity")
ax.set_xlabel("Iterations")
plt.title(r'Convergence of $\frac{\mu}{\Vert\mu\Vert}$ towards $\pm u_1$')

plt.grid(True)
plt.tight_layout()

filename = "plot_softmax_infinite.pdf"
plt.savefig(os.path.join(results, filename),format="pdf", bbox_inches="tight")
plt.close()

# ---------------------------
