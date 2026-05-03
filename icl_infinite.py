#Figure 6b

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
# Parameters
# -----------------------------
n = 10
d = 5
xi = 1
theta = 2
lam = 1e-3
lr = 5e-3
num_steps = 1000
n_runs = 10

xi2 = xi**2

a1 = 2*lam**2 * n*(n+2) * xi2**2 * ((n+d+3)*xi2 + theta)
a2 = lam**2 * n*(n+2) * xi2 * theta * ((3*n + 2*d + 9)*xi2 + (n+5)*theta)
a3 = 2*lam * n * xi2 * ((n+d+1)*xi2 + theta)

b1 = a2
b2 = 2*lam**2 * n*(n+2) * theta**2 * ((2*n + d + 6)*xi2 + (n+4)*theta)
b3 = 2*lam * n * theta * ((2*n + d + 2)*xi2 + (n+2)*theta)

alpha2 = (a3 + b3) / (a1 + 2*a2 + b2)

# -----------------------------
# Spike vector v
# -----------------------------
key = jax.random.PRNGKey(0)
key, sk1 = jax.random.split(key)
v_np = jax.random.normal(sk1, shape=(d,))
v_np /= jnp.linalg.norm(v_np)
v = jnp.array(v_np)

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

# -----------------------------
# Train step
# -----------------------------
@jax.jit
def train_step(mu, v, theta, xi, lam, n, d, lr):
    g = grad_R(mu, v, theta, xi, lam, n, d)
    mu = mu - lr * g
    return mu, g

# -----------------------------
# Logs
# -----------------------------
logs = {"Run": [], "Iterations": [], "Absolute cosine similarity": []}

# -----------------------------
# Multiple runs
# -----------------------------
#key = jax.random.PRNGKey(0)

for run in range(n_runs):
    key, subkey = jax.random.split(key)

    mu = jax.random.normal(subkey, shape=(d,))
    mu = mu / jnp.linalg.norm(mu)

    mu_norm_history = []

    for step in range(num_steps):
        mu, g = train_step(mu, v, theta, xi, lam, n, d, lr)

        mu_normed = mu / jnp.linalg.norm(mu)
        align = jnp.abs(jnp.dot(mu_normed, v))

        mu_norm_history.append(align)

        if step % 1000 == 0:
            print(f"Run {run+1}, Step {step}, alignment: {align:.6f}")

        if jnp.isnan(g).any():
            print(f"NaN detected at step {step}")
            break

    # logging
    for i in range(len(mu_norm_history)):
        logs["Run"].append(run + 1)
        logs["Iterations"].append(i + 1)
        logs["Absolute cosine similarity"].append(float(mu_norm_history[i]))

    print(f"Finished run {run+1}")

# -----------------------------
# Dataframe
# -----------------------------
df = pd.DataFrame(logs)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6), label=r' $|\langle \frac{\mu_k}{\Vert\mu_k\Vert}, v \rangle|$')
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
ax.set_yticks(jnp.arange(0, 1.01, 0.1))
ax.set_ylim(0.1,1.01)

plt.title(r'ICL Infinite prompt: Convergence towards spike direction $\pm v$')

plt.grid(True)
plt.tight_layout()

filename = "plot_ICL_infinite.pdf"
plt.savefig(os.path.join(results, filename), format="pdf", bbox_inches="tight")
plt.close()