import numpy as np
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BENCH_FUNCS = {
    'sphere': (lambda x: -np.sum(x**2), (-5,5)),
    'rastrigin': (lambda x: - (10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))), (-5.12,5.12))
}

def init_population(pop_size, dim, bounds):
    low, high = bounds
    return np.random.uniform(low, high, size=(pop_size, dim))

def fitness(pop, func):
    return np.apply_along_axis(func, 1, pop)

def tournament_selection(pop, fit, k=3):
    idx = np.random.randint(0, len(pop), size=k)
    return pop[idx[np.argmax(fit[idx])]].copy()

def crossover(p1, p2, rate=0.9):
    if np.random.rand() > rate:
        return p1.copy(), p2.copy()
    alpha = np.random.rand()
    return alpha*p1 + (1-alpha)*p2, alpha*p2 + (1-alpha)*p1

def mutate(ind, bounds, m_rate=0.05):
    low, high = bounds
    for i in range(len(ind)):
        if np.random.rand() < m_rate:
            ind[i] += np.random.normal(scale=0.1*(high-low))
            ind[i] = np.clip(ind[i], low, high)
    return ind

def run_ga_core(func, dim=1, bounds=(-5,5), pop_size=50, generations=50, cx_rate=0.9, mut_rate=0.05):
    pop = init_population(pop_size, dim, bounds)
    best_per_gen = []
    best_fit = -np.inf
    best_ind = None
    for g in range(generations):
        fits = fitness(pop, func)
        idx = np.argmax(fits)
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            best_ind = pop[idx].copy()
        best_per_gen.append(best_fit)
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            c1, c2 = crossover(p1, p2, rate=cx_rate)
            c1 = mutate(c1, bounds, m_rate=mut_rate)
            c2 = mutate(c2, bounds, m_rate=mut_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = np.array(new_pop)
    return best_ind, best_fit, best_per_gen

def plot_to_base64(best_per_gen):
    plt.figure(figsize=(6,3))
    plt.plot(best_per_gen, marker='o')
    plt.title('Best fitness per generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='#061827')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def run_and_plot(func='sphere', pop_size=50, generations=50, cx_rate=0.9, mut_rate=0.05):
    f, bounds = BENCH_FUNCS.get(func, BENCH_FUNCS['sphere'])
    best_ind, best_fit, best_per_gen = run_ga_core(lambda x: f(x), dim=1, bounds=bounds,
                                                  pop_size=pop_size, generations=generations,
                                                  cx_rate=cx_rate, mut_rate=mut_rate)
    img_b64 = plot_to_base64(best_per_gen)
    return {'best_ind': best_ind.tolist(), 'best_fit': float(best_fit), 'plot': img_b64}
