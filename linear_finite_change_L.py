#Figure 4b

import matplotlib
matplotlib.use('Agg')
import os
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------
# Setup
# ---------------------------
print("Current working directory:", os.getcwd())
base_dir = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base_dir, "results")
os.makedirs(results, exist_ok=True)

# ---------------------------
# Sampling
# ---------------------------
def sample_batch(key, batch_size, L, Sigma):
    dim = Sigma.shape[0]
    L_ch = jnp.linalg.cholesky(Sigma)
    z = random.normal(key, (batch_size, L, dim), dtype=Sigma.dtype)
    return jnp.matmul(z, L_ch.T)

# ---------------------------
# Loss
# ---------------------------
def batch_loss(mu, X_batch, lam):
    X1 = X_batch[:, 0, :]
    Xk = X_batch[:, :, :]

    alpha = jnp.einsum('bd,d->b', X1, mu)
    betas = jnp.einsum('bkd,d->bk', Xk, mu)

    weighted = jnp.sum(betas[:, :, None] * Xk, axis=1)
    residual = X1 - lam * (alpha[:, None] * weighted)

    return jnp.mean(jnp.sum(residual ** 2, axis=1))

value_and_grad_batch = jit(value_and_grad(batch_loss))

# ---------------------------
# SGD
# ---------------------------
def sgd_minimize(key, Sigma, lam, dim, L, batch_size, lr, num_steps, init_scale):
    key, subkey = random.split(key)
    mu = init_scale * random.normal(subkey, (dim,), dtype=Sigma.dtype)

    for _ in range(num_steps):
        key, subkey = random.split(key)
        X_batch = sample_batch(subkey, batch_size, L, Sigma)

        loss, g = value_and_grad_batch(mu, X_batch, lam)
        mu = mu - lr * g

    return mu

# ---------------------------
# Parameters
# ---------------------------
seed=0
key = random.PRNGKey(seed)

dim = 5
batch_size = 256
lr = 1e-4
num_steps = 5000
lam = 0.001
n_runs = 10

#L_values = jnp.arange(3, 50, 20).astype(int)
L_values = jnp.linspace(3,50, 20).astype(int)




# ---------------------------
# Sigma
# ---------------------------
key, sk1 = random.split(key)
A_full = random.normal(sk1, (dim, dim), dtype=jnp.float32)
Sigma = A_full @ A_full.T + 0.1 * jnp.eye(dim, dtype=jnp.float32)

# ---------------------------
# True top eigenvector
# ---------------------------
vals, vecs = jnp.linalg.eigh(Sigma)
true_top = vecs[:, -1]
true_top = true_top / jnp.linalg.norm(true_top)

# ---------------------------
# Logs
# ---------------------------
logs = {"L": [], "Run": [], "Final cosine similarity": []}

# ---------------------------
# Main experiment
# ---------------------------
for L in L_values:
    print(f"\n=== L = {int(L)} ===")

    for run in range(n_runs):
        key, subkey = random.split(key)

        mu_opt = sgd_minimize(
            subkey,
            Sigma,
            lam,
            dim,
            int(L),
            batch_size,
            lr,
            num_steps,
            init_scale=0.01,
        )

        mu_opt_norm = mu_opt / jnp.linalg.norm(mu_opt)
        cos_sim = jnp.abs(jnp.dot(mu_opt_norm, true_top))

        logs["L"].append(int(L))
        logs["Run"].append(run + 1)
        logs["Final cosine similarity"].append(float(cos_sim))

        print(f"Run {run+1}: {cos_sim:.4f}")

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
    x="L",
    y="Final cosine similarity",
    estimator="mean",
    errorbar ="sd",
    marker="o",
    linewidth=3.0,
)

sns.scatterplot(
    data=df,
    x="L",
    y="Final cosine similarity",
    alpha=0.4,
    legend=False
)

plt.axhline(1, color='red', linestyle='--')

ax.set_ylabel("Final absolute cosine similarity")
ax.set_xlabel("Sequence length L")
ax.set_yticks(jnp.arange(0, 1.01, 0.05))
ax.set_ylim(0.5,1.01)

plt.title(r'Recovery of the top eigenvector vs sequence length $L$')
plt.grid(True)
plt.tight_layout()

filename = "plot_linear_alignment_vs_L_002.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()