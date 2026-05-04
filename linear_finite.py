#Figure 4a

import matplotlib
matplotlib.use('Agg')
import os
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("Current working directory:", os.getcwd())
base_dir = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base_dir, "results")

os.makedirs(results, exist_ok=True)
print("Saving to:", results)

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
    Xk = X_batch[:, 0:, :]

    alpha = jnp.einsum('bd,d->b', X1, mu)
    betas = jnp.einsum('bkd,d->bk', Xk, mu)

    weighted = jnp.sum(betas[:, :, None] * Xk, axis=1)
    residual = X1 - lam * (alpha[:, None] * weighted)

    return jnp.mean(jnp.sum(residual ** 2, axis=1))

value_and_grad_batch = jit(value_and_grad(batch_loss))

def estimate_trace(X_batch):
    return jnp.mean(jnp.sum(X_batch ** 2, axis=-1))

# ---------------------------
# SGD
# ---------------------------
def sgd_minimize(
    key,
    Sigma,
    lam,
    dim,
    L,
    batch_size,
    lr,
    num_steps,
    print_every,
    init_scale,
):
    key, subkey = random.split(key)
    mu = init_scale * random.normal(subkey, (dim,), dtype=Sigma.dtype)

    mu_norm_history = []

    for step in range(1, num_steps + 1):
        key, subkey = random.split(key)
        X_batch = sample_batch(subkey, batch_size, L, Sigma)


        loss, g = value_and_grad_batch(mu, X_batch, lam)
        mu = mu - lr * g

        mu_norm = mu / jnp.linalg.norm(mu)
        mu_norm_history.append(mu_norm)

        if step % print_every == 0 or step == 1:
            print(f"step: {step:6d}  loss: {float(loss):.6e} grad: {jnp.linalg.norm(g)}")

    return mu, mu_norm_history

# ---------------------------
# Parameters
# ---------------------------
key = random.PRNGKey(0)
dim = 5
L = 6
batch_size = 256
lr = 1e-4
num_steps = 5000
lam = 0.01
n_runs = 10

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
logs = {"Run": [], "Iterations": [], "Absolute cosine similarity": []}

# ---------------------------
# Multiple runs
# ---------------------------
for run in range(n_runs):
    key, subkey = random.split(key)

    mu_opt, mu_norm_history = sgd_minimize(
        subkey,
        Sigma,
        lam,
        dim,
        L,
        batch_size,
        lr,
        num_steps,
        print_every=1000,
        init_scale=0.01,
    )

    mu_norm_array = jnp.stack(mu_norm_history)
    sim_history = jnp.abs(jnp.dot(mu_norm_array, true_top))

    # logging
    for i in range(num_steps):
        logs["Run"].append(run + 1)
        logs["Iterations"].append(i + 1)
        logs["Absolute cosine similarity"].append(float(sim_history[i]))

    # diagnostics (your original logic preserved)
    mu_opt_norm = mu_opt / jnp.linalg.norm(mu_opt)
    cos_sim = jnp.abs(jnp.dot(mu_opt_norm, true_top))
    print(f"Run {run+1} final cosine similarity: {cos_sim:.6f}")

# ---------------------------
# Dataframe
# ---------------------------
df = pd.DataFrame(logs)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6), label=r' $|\langle \frac{\mu_k}{\Vert\mu_k\Vert}, u_1 \rangle|$')
sns.set_context("notebook", font_scale=1.5)

ax = sns.lineplot(
    data=df,
    x="Iterations",
    y="Absolute cosine similarity",
    legend=False,
    linewidth=2.0,
    alpha=0.7
)

plt.axhline(1, color='red', linestyle='--')

ax.set_ylabel("Absolute cosine similarity")
ax.set_xlabel("Iterations")
plt.title(r'Convergence of $\frac{\mu}{\Vert\mu\Vert}$ towards $\pm u_1$')

plt.grid(True)
plt.tight_layout()

filename = "plot_linear_attention_finite_L.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()