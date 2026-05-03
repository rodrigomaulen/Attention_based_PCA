# Combined JAX experiment: GD (infinite) vs SGD (finite) in one plot 
# Figure 1

import matplotlib
matplotlib.use('Agg')
import os
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------
# Setup
# ---------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base_dir, "results")
os.makedirs(results, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def normalize(v):
    return v / jnp.linalg.norm(v)

# ---------------------------
# Parameters
# ---------------------------
dim = 5
n_runs = 10
steps = 3000

# infinite (GD)
lambda_inf = 0.01
lr_gd = 1e-3

# finite (SGD)
lam = 0.1
L = 100
batch_size = 256
lr_sgd = 1e-4

# ---------------------------
# Sigma
# ---------------------------
key = random.PRNGKey(0)
key, sk = random.split(key)

A = random.normal(sk, (dim, dim))
Sigma = A @ A.T + 0.1 * jnp.eye(dim)

# ---------------------------
# Top eigenvector
# ---------------------------
eigvals, eigvecs = jnp.linalg.eigh(Sigma)
true_top = eigvecs[:, -1]
true_top = normalize(true_top)

# ---------------------------
# Infinite prompt (GD)
# ---------------------------
def a_of(mu):
    return mu @ (Sigma @ mu)

def b_of(mu):
    return mu @ (Sigma @ Sigma @ mu)

def grad_R(mu):
    a = a_of(mu)
    b = b_of(mu)
    S2 = Sigma @ Sigma

    return -4 * lambda_inf * (S2 @ mu) \
           + 2 * lambda_inf**2 * (a * (S2 @ mu) + b * (Sigma @ mu))

def run_gd(mu_init):
    mu = mu_init
    hist = []

    for _ in range(steps):
        mu = mu - lr_gd * grad_R(mu)
        hist.append(normalize(mu))

    return jnp.stack(hist)

# ---------------------------
# Finite prompt (SGD)
# ---------------------------
def sample_batch(key):
    L_ch = jnp.linalg.cholesky(Sigma)
    z = random.normal(key, (batch_size, L, dim))
    return jnp.matmul(z, L_ch.T)

def loss_fn(mu, X):
    alpha = jnp.einsum('bd,d->b', X[:, 0, :], mu)
    betas = jnp.einsum('bkd,d->bk', X, mu)

    scores = lam * alpha[:, None] * betas
    weights = jax.nn.softmax(scores, axis=1)

    attended = jnp.sum(weights[:, :, None] * X, axis=1)
    residual = X[:, 0, :] - attended

    return jnp.mean(jnp.sum(residual**2, axis=1))

value_and_grad_loss = jit(value_and_grad(loss_fn))

def run_sgd(key):
    key, sk = random.split(key)
    mu = 0.1 * random.normal(sk, (dim,))
    hist = []

    for _ in range(steps):
        key, sk = random.split(key)
        X = sample_batch(sk)

        loss, g = value_and_grad_loss(mu, X)
        mu = mu - lr_sgd * g

        hist.append(normalize(mu))

    return jnp.stack(hist)

# ---------------------------
# Run experiments
# ---------------------------
logs = []

for run in range(n_runs):
    key, sk1, sk2 = random.split(key, 3)

    # GD (infinite)
    mu0 = normalize(random.normal(sk1, (dim,)))
    hist_gd = run_gd(mu0)
    sim_gd = jnp.abs(hist_gd @ true_top)

    # SGD (finite)
    hist_sgd = run_sgd(sk2)
    sim_sgd = jnp.abs(hist_sgd @ true_top)

    for i in range(steps):
        logs.append({
            "Run": run,
            "Iteration": i,
            "Similarity": float(sim_gd[i]),
            "Method": "Infinite prompt (GD)"
        })
        logs.append({
            "Run": run,
            "Iteration": i,
            "Similarity": float(sim_sgd[i]),
            "Method": "Finite prompt (SGD)"
        })

print("Finished all runs")

df = pd.DataFrame(logs)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6))
sns.set_context("notebook", font_scale=1.5)

ax = sns.lineplot(
    data=df,
    x="Iteration",
    y="Similarity",
    hue="Method",
    linewidth=2.5
)

plt.axhline(1, color='black', linestyle='--')

ax.set_ylabel("Absolute cosine similarity")
ax.set_xlabel("Iterations")
plt.title(r'GD (infinite) vs SGD (finite): convergence to $\pm u_1$')

plt.grid(True)
plt.tight_layout()

filename = "comparison_gd_vs_sgd.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()