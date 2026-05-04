# JAX script: alignment vs dimension for analytic risk
#Figure 6a

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
    return v / (jnp.linalg.norm(v) + 1e-8)

# ---------------------------
# Parameters
# ---------------------------
#lam = 0.005
#stepsize = 5e-4
steps = 5000
seed = 0
n_runs = 10
c_lr=0.5
c_lam=0.1

# dimension values (same style as your other script)
d_values = list(range(3, 100, 5))
#d_values=[13]
# or: d_values = [3, 100]

# ---------------------------
# Logs
# ---------------------------
logs = {"Dimension": [], "Run": [], "Final alignment": []}

# ---------------------------
# Main loop over dimension
# ---------------------------
key = random.PRNGKey(seed)

for d in d_values:
    print(f"\n=== d = {d} ===")

    # build Sigma
    key, sk = random.split(key)
    A = random.normal(sk, (d, d))
    Sigma = A @ A.T + 0.1 * jnp.eye(d)

    # precompute matrices
    S = Sigma
    S2 = S @ S

    # ---------------------------
    # gradient
    # ---------------------------
    lam=c_lam/d

    def grad_R(mu):
        a = mu @ (S @ mu)
        b = mu @ (S2 @ mu)

        grad = -4 * lam * (S2 @ mu) \
               + 2 * lam**2 * (a * (S2 @ mu) + b * (S @ mu))
        return grad

    # ---------------------------
    # GD
    # ---------------------------
    def run_gd(mu_init):
        mu = mu_init
        for _ in range(steps):
            g = grad_R(mu)
            lr=c_lr/d**2
            mu = mu - lr * g
        return mu

    # ---------------------------
    # top eigenvector
    # ---------------------------
    eigvals, eigvecs = jnp.linalg.eigh(Sigma)
    true_top = eigvecs[:, -1]
    true_top = normalize(true_top)

    # ---------------------------
    # runs
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

plt.title("Analytic risk: recovery vs dimension")

plt.grid(True)
plt.tight_layout()

filename = "plot_alignment_vs_dimension_analytic_lam_lr.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()