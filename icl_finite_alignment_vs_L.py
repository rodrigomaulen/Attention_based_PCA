#Figure 7a

import matplotlib
matplotlib.use('Agg')
import os
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("Current working directory:", os.getcwd())
base_dir = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base_dir, "results")

os.makedirs(results, exist_ok=True)
print("Saving to:", results)

key = jax.random.PRNGKey(0)

# -------------------------
# Parameters
# -------------------------
d = 5
n = 10
xi = 1.0
theta = 2.0
lam = 1e-3
steps = 1000
lr = 5e-3
N = 100
n_sigma_mc = N
n_x_mc = N
n_runs = 10

# L values
L_values = list(range(3, 101, 5))

# -------------------------
# Spike vector v
# -------------------------
v_np = jax.random.normal(key, (d,))
v_np = v_np / jnp.linalg.norm(v_np)
v = jnp.array(v_np)

# -------------------------
# Utilities
# -------------------------
def generate_wishart(key, V, df, batch):
    d = V.shape[0]
    keys = jax.random.split(key, batch)
    def sample_one(k):
        Z = jax.random.multivariate_normal(k, jnp.zeros(d), V, (df,))
        return Z.T @ Z
    return vmap(sample_one)(keys)

def generate_X_given_Sigma(key, Sigma, n_samples, L):
    keys = jax.random.split(key, n_samples)
    def sample_one(k):
        return jax.random.multivariate_normal(k, jnp.zeros(d), Sigma, (L,))
    return vmap(sample_one)(keys)

# -------------------------
# Soft attention
# -------------------------
def T_soft(mu, X_batch):
    def per_sample(X):
        scores = lam * (X @ mu) * (X[0] @ mu)
        w = jax.nn.softmax(scores)
        return jnp.sum(w[:, None] * X, axis=0)
    return vmap(per_sample)(X_batch)

# -------------------------
# Risk
# -------------------------
def R_soft_L(mu, X_batch):
    T1 = T_soft(mu, X_batch)
    X1 = X_batch[:, 0, :]
    return jnp.mean(jnp.sum((X1 - T1)**2, axis=1))

def R_ICL(mu, key, L):
    V = xi**2 * jnp.eye(d) + theta * jnp.outer(v, v)

    key_sigma, key_x = jax.random.split(key)
    Sigmas = generate_wishart(key_sigma, V, n, n_sigma_mc)
    keys_x = jax.random.split(key_x, n_sigma_mc)

    def per_sigma(Sigma, kx):
        X_batch = generate_X_given_Sigma(kx, Sigma, n_x_mc, L)
        return R_soft_L(mu, X_batch)

    return jnp.mean(vmap(per_sigma)(Sigmas, keys_x))

# -------------------------
# Gradient
# -------------------------
grad_R_ICL = jit(grad(R_ICL, argnums=0), static_argnums=2)

# -------------------------
# Logs
# -------------------------
logs = {"L": [], "Run": [], "Final alignment": []}

# -------------------------
# Main loop over L
# -------------------------
for L in L_values:
    print(f"\n=== L = {L} ===")

    for run in range(n_runs):
        key, subkey = jax.random.split(key)

        mu = jax.random.normal(subkey, (d,))
        mu = mu / jnp.linalg.norm(mu)

        for t in range(steps):
            key, subkey = jax.random.split(key)
            g = grad_R_ICL(mu, subkey, L)
            mu = mu - lr * g

        # final alignment
        mu_normed = mu / jnp.linalg.norm(mu)
        align = jnp.abs(jnp.dot(mu_normed, v))

        logs["L"].append(L)
        logs["Run"].append(run + 1)
        logs["Final alignment"].append(float(align))

        print(f"Run {run+1}: alignment = {align:.4f}")

# -------------------------
# Dataframe
# -------------------------
df = pd.DataFrame(logs)

# -------------------------
# Plot
# -------------------------
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
plt.title("ICL Finite: Recovery vs context length L")

plt.grid(True)
plt.tight_layout()

filename = "plot_ICL_finite_vs_L.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()