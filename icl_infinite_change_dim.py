# JAX script: ICL infinite — alignment vs dimension
#Figure 7b

import matplotlib
matplotlib.use('Agg')
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("Current working directory:", os.getcwd())
base_dir = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base_dir, "results")

os.makedirs(results, exist_ok=True)
print("Saving to:", results)

# -----------------------------
# Fixed parameters
# -----------------------------
#n = 50
xi = 1
theta = 2
num_steps = 2000
n_runs = 10

# scaling constants
c_lam = 0.1
c_lr = 0.5

# dimensions
d_values = list(range(3, 100, 5))
#d_values = [3, 100]

# -----------------------------
# Logs
# -----------------------------
logs = {"Dimension": [], "Run": [], "Final alignment": []}

# -----------------------------
# Risk
# -----------------------------
def R_ICL(mu, v, theta, xi, lam, n, d):
    r2 = jnp.dot(mu, mu)
    alpha2_loc = jnp.dot(mu, v) ** 2

    term0 = n * (d * xi**2 + theta)

    A = xi**2 * r2 + theta * alpha2_loc
    B = xi**4 * r2 + (2 * xi**2 * theta + theta**2) * alpha2_loc

    term1 = -2 * lam * n * (
        (n + 1) * B
        + (d * xi**2 + theta) * A
    )

    term2 = lam**2 * n * (n + 2) * (
        (n + 3) * A * B
        + (d * xi**2 + theta) * A**2
    )

    return term0 + term1 + term2

grad_R = jax.grad(R_ICL, argnums=0)

@jax.jit
def train_step(mu, v, theta, xi, lam, n, d, lr):
    g = grad_R(mu, v, theta, xi, lam, n, d)
    mu = mu - lr * g
    return mu

# -----------------------------
# Main loop over dimension
# -----------------------------
key = jax.random.PRNGKey(0)

for d in d_values:
    print(f"\n=== d = {d} ===")

    # scaling
    n=d
    lam= c_lam / d
    lr = c_lr / (d**2)
    #lr= c_lr/(d*n)

    # spike vector
    key, sk = jax.random.split(key)
    v = jax.random.normal(sk, shape=(d,))
    v = v / jnp.linalg.norm(v)

    for run in range(n_runs):
        key, subkey = jax.random.split(key)

        mu = jax.random.normal(subkey, shape=(d,))
        mu = mu / jnp.linalg.norm(mu)

        # GD
        for _ in range(num_steps):
            mu = train_step(mu, v, theta, xi, lam, n, d, lr)

        mu_norm = mu / jnp.linalg.norm(mu)
        alignment = jnp.abs(jnp.dot(mu_norm, v))

        logs["Dimension"].append(d)
        logs["Run"].append(run + 1)
        logs["Final alignment"].append(float(alignment))

        print(f"Run {run+1}: alignment = {alignment:.4f}")

# -----------------------------
# Dataframe
# -----------------------------
df = pd.DataFrame(logs)

# -----------------------------
# Plot
# -----------------------------
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
ax.set_yticks(jnp.arange(0, 1.01, 0.1))
ax.set_ylim(0.1,1.01)

plt.title("ICL infinite prompt: recovery vs dimension")

plt.grid(True)
plt.tight_layout()

filename = "plot_ICL_infinite_vs_dimension_test.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()