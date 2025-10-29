import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def make_dataset(n=200, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 10, size=(n,1))
    y = 10 + 8*X.flatten() - 0.6*(X.flatten()**2) + rng.normal(scale=4.0, size=n)
    y = np.clip(y, 0, 100)
    return X, y

def train_and_plot(size=200, seed=1):
    X, y = make_dataset(n=size, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = MLPRegressor(hidden_layer_sizes=(16,8), max_iter=500, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    idx = np.argsort(X_test.flatten())
    x_line = X_test.flatten()[idx].tolist()
    y_line = y_pred[idx].tolist()
    scatter_x = X_test.flatten().tolist()
    scatter_y = y_test.tolist()

    plot_data = {
        'line_x': x_line,
        'line_y': y_line,
        'scatter_x': scatter_x,
        'scatter_y': scatter_y
    }
    metrics = {'mse': float(mse)}
    return plot_data, metrics
