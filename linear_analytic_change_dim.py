# JAX script: alignment vs dimension for R_lin,L
# Figure 5b

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
#L = 50
steps = 5000
n_runs = 10
seed = 0

# scaling constants (important!)
c_lam = 0.01
c_lr = 1

# dimensions (5 points between 3 and 100)
d_values = jnp.arange(3, 101, 5)
#d_values=[3,100]

# ---------------------------
# Logs
# ---------------------------
logs = {"Dimension": [], "Run": [], "Final alignment": []}

# ---------------------------
# Main loop
# ---------------------------
key = random.PRNGKey(seed)

for d in d_values:
    d = int(d)
    print(f"\n=== d = {d} ===")

    # scale parameters with dimension
    L=d
    lam = c_lam / d
    stepsize = c_lr / d**2

    # build Sigma
    key, sk = random.split(key)
    A = random.normal(sk, (d, d))
    Sigma = A @ A.T + 0.1 * jnp.eye(d)

    TrSigma = jnp.trace(Sigma)
    S2 = Sigma @ Sigma

    # ---------------------------
    # Risk
    # ---------------------------
    def R_lin(mu):
        a = mu @ (Sigma @ mu)
        b = mu @ (S2 @ mu)

        return (
            TrSigma
            - 2 * lam * TrSigma * a
            - 2 * lam * (L + 1) * b
            + lam**2 * (L + 2) * TrSigma * a**2
            + lam**2 * (L + 2) * (L + 3) * a * b
        )

    grad_R = jit(grad(R_lin))

    # ---------------------------
    # GD
    # ---------------------------
    def run_gd(mu_init):
        mu = mu_init
        for _ in range(steps):
            g = grad_R(mu)
            mu = mu - stepsize * g
        return mu

    # ---------------------------
    # Top eigenvector
    # ---------------------------
    eigvals, eigvecs = jnp.linalg.eigh(Sigma)
    true_top = normalize(eigvecs[:, -1])

    # ---------------------------
    # Runs
    # ---------------------------
    for run in range(n_runs):
        key, subkey = random.split(key)

        mu_init = random.normal(subkey, (d,))
        mu_init = normalize(mu_init)

        mu_star = run_gd(mu_init)

        mu_norm = normalize(mu_star)
        alignment = jnp.abs(jnp.dot(mu_norm, true_top))

        logs["Dimension"].append(d)
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
    x="Dimension",
    y="Final alignment",
    estimator="mean",
    errorbar="sd",
    linewidth=3.0,
)

sns.scatterplot(
    data=df,
    x="Dimension",
    y="Final alignment",
    alpha=0.4,
    legend=False
)

plt.axhline(1, color='red', linestyle='--')

ax.set_ylabel("Final absolute cosine similarity")
ax.set_xlabel("Dimension d")
ax.set_yticks(jnp.arange(0, 1.01, 0.01))
ax.set_ylim(0.94,1.01)

plt.title("Linear risk: recovery vs dimension")

plt.grid(True)
plt.tight_layout()

filename = "plot_linear_analytic_vs_dimension.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()