import numpy as np
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def obj(x):
    return -np.sum(x**2)

def run_pso(n_particles=30, iterations=60, dim=1, bounds=(-5,5)):
    low, high = bounds
    rng = np.random.RandomState()
    X = rng.uniform(low, high, size=(n_particles, dim))
    V = rng.uniform(-(high-low)*0.1, (high-low)*0.1, size=(n_particles, dim))
    pbest = X.copy()
    pbest_val = np.apply_along_axis(obj, 1, pbest)
    gbest = pbest[np.argmax(pbest_val)].copy()
    gbest_val = np.max(pbest_val)
    gbest_history = []
    for t in range(iterations):
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        for i in range(n_particles):
            r1, r2 = rng.rand(), rng.rand()
            V[i] = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(gbest - X[i])
            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i], low, high)
            val = obj(X[i])
            if val > pbest_val[i]:
                pbest[i] = X[i].copy()
                pbest_val[i] = val
                if val > gbest_val:
                    gbest = X[i].copy()
                    gbest_val = val
        gbest_history.append(gbest_val)
    return gbest, gbest_val, gbest_history

def plot_to_base64(history):
    plt.figure(figsize=(6,3))
    plt.plot(history)
    plt.title('Global Best over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='#061827')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def run_and_plot(n_particles=30, iterations=60):
    gbest, gbest_val, history = run_pso(n_particles=n_particles, iterations=iterations)
    img = plot_to_base64(history)
    return {'gbest': gbest.tolist(), 'gbest_val': float(gbest_val), 'plot': img}
