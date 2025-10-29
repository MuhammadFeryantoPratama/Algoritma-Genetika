from flask import Flask, render_template, request
from modules import fuzzy_module, ann_module, ga_module, pso_module, bayes_module

main = Flask(__name__)

@main.route('/')
def index():
    return render_template('index.html')

# Fuzzy
@main.route('/fuzzy', methods=['GET', 'POST'])
def fuzzy():
    result = None
    if request.method == 'POST':
        temp = float(request.form.get('temperature', 25))
        result = fuzzy_module.evaluate_temperature(temp)
    return render_template('fuzzy.html', result=result)

# ANN
@main.route('/ann', methods=['GET', 'POST'])
def ann():
    plot_data = None
    metrics = None
    if request.method == 'POST':
        size = int(request.form.get('size', 200))
        seed = int(request.form.get('seed', 1))
        plot_data, metrics = ann_module.train_and_plot(size=size, seed=seed)
    return render_template('ann.html', plot_data=plot_data, metrics=metrics)

# GA
@main.route('/ga', methods=['GET', 'POST'])
def ga():
    result = None
    if request.method == 'POST':
        pop = int(request.form.get('pop', 50))
        gens = int(request.form.get('gens', 60))
        cx = float(request.form.get('cx', 0.9))
        mut = float(request.form.get('mut', 0.05))
        func = request.form.get('func', 'sphere')
        result = ga_module.run_and_plot(func=func, pop_size=pop, generations=gens, cx_rate=cx, mut_rate=mut)
    return render_template('ga.html', result=result)

# PSO
@main.route('/pso', methods=['GET', 'POST'])
def pso():
    result = None
    if request.method == 'POST':
        n_particles = int(request.form.get('n_particles', 30))
        iters = int(request.form.get('iters', 60))
        result = pso_module.run_and_plot(n_particles=n_particles, iterations=iters)
    return render_template('pso.html', result=result)

# Bayesian
@main.route('/bayes', methods=['GET', 'POST'])
def bayes():
    output = None
    if request.method == 'POST':
        fever = request.form.get('fever') == 'on'
        cough = request.form.get('cough') == 'on'
        fatigue = request.form.get('fatigue') == 'on'
        output = bayes_module.diagnose({'fever': fever, 'cough': cough, 'fatigue': fatigue})
    return render_template('bayes.html', output=output)

if __name__ == '__main__':
    main.run(debug=True)
