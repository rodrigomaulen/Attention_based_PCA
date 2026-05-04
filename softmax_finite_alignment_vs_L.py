# JAX experiment: performance softmax vs sequence length L
#Figure 5a

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
# Paths
# ---------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base_dir, "results")
os.makedirs(results, exist_ok=True)

# ---------------------------
# Sampling
# ---------------------------
def sample_batch(key, batch_size, L_seq, Sigma):
    dim = Sigma.shape[0]
    L_ch = jnp.linalg.cholesky(Sigma)
    z = random.normal(key, (batch_size, L_seq, dim), dtype=Sigma.dtype)
    return jnp.matmul(z, L_ch.T)

# ---------------------------
# Loss
# ---------------------------
def batch_loss_softmax(mu, X_batch, lam):
    Xk = X_batch

    alpha = jnp.einsum('bd,d->b', X_batch[:, 0, :], mu)
    betas = jnp.einsum('bkd,d->bk', Xk, mu)

    scores = lam * alpha[:, None] * betas
    weights = jax.nn.softmax(scores, axis=1)

    attended = jnp.sum(weights[:, :, None] * Xk, axis=1)
    residual = X_batch[:, 0, :] - attended

    return jnp.mean(jnp.sum(residual ** 2, axis=1))

value_and_grad_batch = jit(value_and_grad(batch_loss_softmax))

# ---------------------------
# SGD
# ---------------------------
def sgd_minimize(key, Sigma, lam, dim, L_seq, batch_size, lr, num_steps):
    key, subkey = random.split(key)
    mu = 0.1 * random.normal(subkey, (dim,), dtype=Sigma.dtype)

    for _ in range(num_steps):
        key, subkey = random.split(key)
        X_batch = sample_batch(subkey, batch_size, L_seq, Sigma)
        loss, g = value_and_grad_batch(mu, X_batch, lam)
        mu = mu - lr * g

    return mu

# ---------------------------
# Parameters
# ---------------------------
key = random.PRNGKey(0)

dim = 5
batch_size = 256
lr = 1e-4
num_steps = 5000
lam = 0.1
n_runs = 10

# L values to test
#L_values = list(range(3, 101, 5))  
L_values=jnp.linspace(3,50,20).astype(int)

# ---------------------------
# Sigma
# ---------------------------
key, sk1, sk2 = random.split(key, 3)
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
logs = {"L": [], "Run": [], "Final alignment": []}

# ---------------------------
# Experiment loop
# ---------------------------
for L_seq in L_values:
    print(f"\n=== L = {L_seq} ===")

    for run in range(n_runs):
        key, subkey = random.split(key)

        mu_opt = sgd_minimize(
            subkey,
            Sigma,
            lam,
            dim,
            L_seq,
            batch_size,
            lr,
            num_steps,
        )

        mu_norm = mu_opt / jnp.linalg.norm(mu_opt)
        alignment = jnp.abs(jnp.dot(mu_norm, true_top))

        logs["L"].append(int(L_seq))
        logs["Run"].append(run + 1)
        logs["Final alignment"].append(float(alignment))

        print(f"Run {run+1}: alignment = {alignment:.4f}")

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
    y="Final alignment",
    estimator="mean",
    errorbar="sd",
    linewidth=3.0,
)

# scatter points (optional, nice for paper)
sns.scatterplot(
    data=df,
    x="L",
    y="Final alignment",
    alpha=0.4,
    legend=False
)

plt.axhline(1, color='red', linestyle='--')

ax.set_ylabel("Final absolute cosine similarity")
ax.set_xlabel("Sequence length L")
ax.set_yticks(jnp.arange(0, 1.01, 0.05))
ax.set_ylim(0.5,1.01)

plt.title("Recovery of top eigenvector vs sequence length L")

plt.grid(True)
plt.tight_layout()

filename = "plot_alignment_vs_L_01.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()