import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import scipy.special as sp
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import time


class LNLM:
    def __init__(self, k_folds=5, degree=4):
        self.k_folds = k_folds
        self.mu = None
        self.degree = degree
        self.non_linear_model = None
        self.linear_model = LinearRegression()

    def fit(self, X, y):
        # Generate Hermite polynomial features for the entire dataset
        herm = sp.hermitenorm(self.degree, monic=True)
        X_hermite = herm(X)

        # Add constant term to X_hermite
        X_hermite = np.concatenate(
            (np.ones((X_hermite.shape[0], 1)), X_hermite), axis=1
        )

        # Step 1: Choose the value of μ using k-fold cross-validation
        kf = KFold(n_splits=self.k_folds)
        min_rmse = float("inf")
        best_mu = None

        total_time = 0

        for mu in np.linspace(0, 1, 100):
            fold_rmse_sum = 0
            start_time = time.time()
            for train_index, val_index in kf.split(X):
                X_train, X_val = X_hermite[train_index], X_hermite[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Fit the linear model and non-linear model separately
                linear_model = LinearRegression()
                linear_model.fit(X_train, y_train)

                non_linear_model = self.fit_non_linear(X_train, y_train)

                # Predict using the current μ
                y_pred = mu * non_linear_model.predict(X_val) + (
                    1 - mu
                ) * linear_model.predict(X_val)

                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
                fold_rmse_sum += rmse

            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time

            avg_rmse = fold_rmse_sum / self.k_folds
            if avg_rmse < min_rmse:
                min_rmse = avg_rmse
                best_mu = mu

        self.mu = best_mu

        # Step 2: Fit the non-linear model and linear model using full data
        start_time = time.time()
        self.non_linear_model = self.fit_non_linear(X_hermite, y)
        self.linear_model.fit(X_hermite, y)
        end_time = time.time()
        total_time += end_time - start_time

        return total_time

    def predict(self, X):
        # Generate Hermite polynomial features for prediction
        herm = sp.hermitenorm(self.degree, monic=True)
        X_hermite = herm(X)

        # Add constant term to X_hermite
        X_hermite = np.concatenate(
            (np.ones((X_hermite.shape[0], 1)), X_hermite), axis=1
        )

        # Combine predictions from linear and non-linear models using the chosen μ
        linear_pred = self.linear_model.predict(X_hermite)
        non_linear_pred = self.predict_non_linear(X_hermite)
        return self.mu * non_linear_pred + (1 - self.mu) * linear_pred

    def fit_non_linear(self, X, y):
        # Fit the non-linear model
        non_linear_model = LinearRegression()
        non_linear_model.fit(X, y)
        return non_linear_model

    def predict_non_linear(self, X):
        # Predict using the non-linear model
        return self.non_linear_model.predict(X)


def generate_data(num_points, func):
    # Generate X from a Student’s t distribution with 4 degrees of freedom
    X = np.random.standard_t(df=4, size=num_points)

    # Generate noisy Y values using a function
    Y = generate_Y(X, func)

    return X.reshape(-1, 1), Y


def generate_Y(X, func):
    # Define the reaction function φ(X)
    phi_X = func(X)

    # Add noise term
    epsilon = np.random.standard_t(df=4, size=X.shape[0])
    Y = phi_X + epsilon

    return Y


def evaluate_model(X_train, Y_train, X_test, Y_test, degree):
    # Fit LNLM model
    model = LNLM(degree=degree)
    total_time = model.fit(X_train, Y_train)

    # Predict using the model
    start_time = time.time()
    Y_pred = model.predict(X_test)
    end_time = time.time()
    total_time += end_time - start_time

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

    return rmse, total_time


def run_simulation(funcs, degrees, num_points_list, num_simulations=1000):
    results = []

    for func_name, func in funcs.items():
        for degree in degrees:
            for num_points in num_points_list:
                mse_sum = 0
                time_sum = 0

                for sim in range(num_simulations):
                    # Generate data
                    X_train, Y_train = generate_data(num_points, func)
                    X_test, Y_test = generate_data(num_points, func)

                    # Evaluate model
                    mse, total_time = evaluate_model(
                        X_train, Y_train, X_test, Y_test, degree
                    )
                    mse_sum += mse
                    time_sum += total_time

                    # Write ave results to CSV
                    with open(
                        "result_for_"
                        + func_name
                        + "_"
                        + str(num_points)
                        + "_"
                        + str(sim)
                        + "_.csv",
                        "w",
                        newline="",
                    ) as csvfile:
                        fieldnames = ["function", "degree", "num_pts", "mse", "time"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(
                            {
                                "function": func_name,
                                "degree": degree,
                                "num_pts": num_points,
                                "mse": mse,
                                "time": total_time,
                            }
                        )

                # Calculate average RMSE and time
                avg_mse = mse_sum / num_simulations
                avg_time = time_sum / num_simulations

                # Record results
                results.append((func_name, degree, num_points, avg_mse, avg_time))

    return results


# Define target functions φ(X)
def phi315(x):
    return 0.33 * x


def phi316(x):
    return 0.8 + 0.8 * x


def phi317(x):
    return -2 + 0.75 * x + 0.2 * x**2


def phi318(x):
    return 2 + np.cos(x / 2) + 0.5 * x


def phi319(x):
    return 0.01 * np.exp(x) - 0.1 * x**2


def phi320(x):
    return 0.1 + 0.1 * x + 0.02 * x**2 + 0.03 * x**3


def phi321(x):
    return 0.1 + 0.1 * np.sin(x) - 0.3 * x


def phi322(x):
    return -3 - 0.5 * x + 0.05 * x**2


def phi323(x):
    return 0.1 - 0.01 * x + 0.002 * x**2 - 0.001 * x**3 + 0.001 * x**4


def phi324(x):
    return 3 + np.tanh(x) + 0.5 * x


def phi325(x):
    return -0.4 + 0.5 * np.abs(x)


def phi326(x):
    return 0.5 * np.sinh(0.01 * x) - 0.005 * x**3


def phi327(x):
    return 3


# Create dictionary of target functions
funcs = {
    "phi315": phi315,
    "phi316": phi316,
    "phi317": phi317,
    "phi318": phi318,
    "phi319": phi319,
    "phi320": phi320,
    "phi321": phi321,
    "phi322": phi322,
    "phi323": phi323,
    "phi324": phi324,
    "phi325": phi325,
    "phi326": phi326,
    "phi327": phi327,
}

num_points_list = [126, 252, 756, 1260]

# Run evaluation
results = run_simulation(funcs, [4], num_points_list)

# Write results to CSV
with open("all_results.csv", "w", newline="") as csvfile:
    fieldnames = ["function", "degree", "num_pts", "avg_rsme", "avg_time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "function": result[0],
                "degree": result[1],
                "num_pts": result[2],
                "avg_rsme": result[3],
                "avg_time": result[4],
            }
        )
