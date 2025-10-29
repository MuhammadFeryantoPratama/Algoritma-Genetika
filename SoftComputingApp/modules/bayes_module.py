# Lightweight Bayes diagnoser
PRIORS = {
    'Flu': 0.05,
    'Cold': 0.10,
    'Healthy': 0.85
}
LIKELIHOODS = {
    'Flu': {'fever': 0.9, 'cough': 0.7, 'fatigue': 0.8},
    'Cold': {'fever': 0.3, 'cough': 0.8, 'fatigue': 0.4},
    'Healthy': {'fever': 0.01, 'cough': 0.05, 'fatigue': 0.02}
}

def diagnose(symptoms: dict):
    post_unnorm = {}
    for d in PRIORS:
        p = PRIORS[d]
        for s, present in symptoms.items():
            lik = LIKELIHOODS[d].get(s, 0.01)
            p *= (lik if present else (1 - lik))
        post_unnorm[d] = p
    total = sum(post_unnorm.values())
    if total == 0:
        return {d: 0.0 for d in PRIORS}
    posterior = {d: round(post_unnorm[d] / total, 4) for d in PRIORS}
    return posterior
