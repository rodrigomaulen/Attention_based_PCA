# JAX softmax SGD experiment with multiple runs + seaborn plot
#Figure 2a

import matplotlib
matplotlib.use('Agg')
import os
import jax
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
    X = jnp.matmul(z, L_ch.T)
    return X

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

    loss = jnp.mean(jnp.sum(residual ** 2, axis=1))
    return loss

value_and_grad_batch = jit(value_and_grad(batch_loss_softmax))

# ---------------------------
# SGD
# ---------------------------
def sgd_minimize(
    key,
    Sigma,
    lam=1,
    dim=None,
    L=6,
    batch_size=256,
    lr=1e-3,
    num_steps=2000,
    print_every=200,
    init_scale=0.1,
):
    if dim is None:
        dim = Sigma.shape[0]

    key, subkey = random.split(key)
    mu = init_scale * random.normal(subkey, (dim,), dtype=Sigma.dtype)

    mu_norm_history = []
    loss_history = []
    trace_avg = 0.0

    for step in range(1, num_steps + 1):
        key, subkey = random.split(key)
        X_batch = sample_batch(subkey, batch_size, L, Sigma)

        loss, g = value_and_grad_batch(mu, X_batch, lam)
        mu = mu - lr * g

        mu_norm = mu / jnp.linalg.norm(mu)
        mu_norm_history.append(mu_norm)

        if step % print_every == 0 or step == 1:
            loss_val = float(loss)
            loss_history.append((step, loss_val))
            print(f"step: {step:6d}  loss: {loss_val:.6e}")

    return mu, loss_history, trace_avg, mu_norm_history

# ---------------------------
# Parameters
# ---------------------------
key = random.PRNGKey(0)
dim = 5
L = 100
batch_size = 256
lr = 1e-4
num_steps = 3000
lam = 0.1
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
top_idx = jnp.argmax(vals)
true_top = vecs[:, top_idx]
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

    mu_opt, history, trace_avg, mu_norm_history = sgd_minimize(
        subkey,
        Sigma,
        lam=lam,
        dim=dim,
        L=L,
        batch_size=batch_size,
        lr=lr,
        num_steps=num_steps,
        print_every=1000,
        init_scale=0.1,
    )

    mu_norm_array = jnp.stack(mu_norm_history)
    sim_history = jnp.abs(jnp.dot(mu_norm_array, true_top))

    for i in range(num_steps):
        logs["Run"].append(run + 1)
        logs["Iterations"].append(i + 1)
        logs["Absolute cosine similarity"].append(float(sim_history[i]))

    # final diagnostics (same as your original)
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
    linewidth=3.0,
    alpha=0.7
)

plt.axhline(1, color='red', linestyle='--')

ax.set_ylabel("Absolute cosine similarity")
ax.set_xlabel("Iterations")
plt.title(r'Convergence of $\frac{\mu}{\Vert\mu\Vert}$ towards $\pm u_1$')

plt.grid(True)
plt.tight_layout()

filename = "plot_softmax_finite_L.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()