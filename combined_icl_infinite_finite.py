# Combined ICL experiment: GD (infinite) vs SGD (finite)

import matplotlib
matplotlib.use('Agg')
import os
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
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
# Parameters
# ---------------------------
d = 5
n = 10
xi = 1.0
theta = 2.0
lam = 1e-3
lr = 5e-3
steps = 1000
n_runs = 10

# finite prompt
L = 100
N = 100
n_sigma_mc = N
n_x_mc = N

key = jax.random.PRNGKey(0)

# ---------------------------
# Spike vector v
# ---------------------------
key, sk = jax.random.split(key)
v = jax.random.normal(sk, (d,))
v = v / jnp.linalg.norm(v)

# ===========================
# ===== INFINITE (GD) =======
# ===========================

def R_ICL_infinite(mu):
    r2 = jnp.dot(mu, mu)
    alpha2 = jnp.dot(mu, v) ** 2

    term0 = n * (d * xi**2 + theta)

    A = xi**2 * r2 + theta * alpha2
    B = xi**4 * r2 + (2 * xi**2 * theta + theta**2) * alpha2

    term1 = -2 * lam * n * ((n + 1) * B + (d * xi**2 + theta) * A)

    term2 = lam**2 * n * (n + 2) * (
        (n + 3) * A * B + (d * xi**2 + theta) * A**2
    )

    return term0 + term1 + term2

grad_inf = jit(grad(R_ICL_infinite))

def run_gd(mu_init):
    mu = mu_init
    hist = []

    for _ in range(steps):
        mu = mu - lr * grad_inf(mu)
        mu_norm = mu / jnp.linalg.norm(mu)
        hist.append(jnp.abs(jnp.dot(mu_norm, v)))

    return jnp.array(hist)

# ===========================
# ===== FINITE (SGD) ========
# ===========================

def generate_wishart(key, V, df, batch):
    d = V.shape[0]
    keys = jax.random.split(key, batch)

    def sample_one(k):
        Z = jax.random.multivariate_normal(k, jnp.zeros(d), V, (df,))
        return Z.T @ Z

    return vmap(sample_one)(keys)

def generate_X_given_Sigma(key, Sigma, n_samples):
    keys = jax.random.split(key, n_samples)

    def sample_one(k):
        return jax.random.multivariate_normal(k, jnp.zeros(d), Sigma, (L,))

    return vmap(sample_one)(keys)

def T_soft(mu, X_batch):
    def per_sample(X):
        scores = lam * (X @ mu) * (X[0] @ mu)
        w = jax.nn.softmax(scores)
        return jnp.sum(w[:, None] * X, axis=0)

    return vmap(per_sample)(X_batch)

def R_soft_L(mu, X_batch):
    T1 = T_soft(mu, X_batch)
    X1 = X_batch[:, 0, :]
    return jnp.mean(jnp.sum((X1 - T1)**2, axis=1))

def R_ICL_finite(mu, key):
    V = xi**2 * jnp.eye(d) + theta * jnp.outer(v, v)

    key_sigma, key_x = jax.random.split(key)
    Sigmas = generate_wishart(key_sigma, V, n, n_sigma_mc)
    keys_x = jax.random.split(key_x, n_sigma_mc)

    def per_sigma(Sigma, kx):
        X_batch = generate_X_given_Sigma(kx, Sigma, n_x_mc)
        return R_soft_L(mu, X_batch)

    return jnp.mean(vmap(per_sigma)(Sigmas, keys_x))

grad_fin = jit(grad(R_ICL_finite))

def run_sgd(key):
    key, sk = jax.random.split(key)
    mu = jax.random.normal(sk, (d,))
    mu = mu / jnp.linalg.norm(mu)

    hist = []

    for _ in range(steps):
        key, sk = jax.random.split(key)
        g = grad_fin(mu, sk)
        mu = mu - lr * g

        mu_norm = mu / jnp.linalg.norm(mu)
        hist.append(jnp.abs(jnp.dot(mu_norm, v)))

    return jnp.array(hist)

# ===========================
# ===== RUN EXPERIMENT ======
# ===========================

logs = []

for run in range(n_runs):
    print(run)
    key, k1, k2 = jax.random.split(key, 3)

    # infinite (GD)
    mu0 = jax.random.normal(k1, (d,))
    mu0 = mu0 / jnp.linalg.norm(mu0)
    hist_gd = run_gd(mu0)

    # finite (SGD)
    hist_sgd = run_sgd(k2)

    for i in range(steps):
        logs.append({
            "Run": run,
            "Iteration": i,
            "Similarity": float(hist_gd[i]),
            "Method": "Infinite prompt (GD)"
        })
        logs.append({
            "Run": run,
            "Iteration": i,
            "Similarity": float(hist_sgd[i]),
            "Method": "Finite prompt (SGD)"
        })

print("Finished all runs")

df = pd.DataFrame(logs)

# ===========================
# ===== PLOT ================
# ===========================

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
ax.set_yticks(jnp.arange(0, 1.01, 0.1))
ax.set_ylim(0.1, 1.01)

plt.title(r'ICL: GD (infinite) vs SGD (finite) — alignment to $\pm v$')

plt.grid(True)
plt.tight_layout()

filename = "comparison_ICL_gd_vs_sgd.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()