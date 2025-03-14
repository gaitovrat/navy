import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(
        self, *, learning_rate=0.1, n_iterations=1000, random_state: int | None = None
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.zeros(0)
        self.bias = 0

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit perception weights and bias"""
        n_features = X.shape[1]
        self.weights = np.random.randn(n_features)
        self.bias = np.random.random()

        for iteration in range(self.n_iterations):
            was_error = False

            for idx, x_i in enumerate(X):
                y_pred = self.activation(x_i)
                y_true = y[idx]

                # If the same sign then prediction is correct
                if y_true * y_pred <= 0:
                    was_error = True
                    self.weights += self.learning_rate * y_true * x_i
                    self.bias += self.learning_rate * y_true

            if not was_error:
                print(f"Ending early on iteration: {iteration}")
                break

    def activation(self, x: np.ndarray) -> np.ndarray:
        """Linear activation: w · x + b"""
        return np.dot(x, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class based on sign of activation"""
        return np.sign(np.dot(X, self.weights) + self.bias)


def main() -> None:
    perceptron = Perceptron()
    n = 100
    min_value = -10
    max_value = 10

    X = np.random.uniform(min_value, max_value, (n, 2))
    y = np.zeros(n)

    x_coords = X[:, 0]
    y_coords = X[:, 1]

    y_line = 3 * x_coords + 2
    diff = y_coords - y_line
    y[diff > 0] = 1
    y[diff < 0] = -1

    perceptron.fit(X, y)
    y_pred = perceptron.predict(X)

    plt.figure(figsize=(10, 8))

    plt.scatter(
        X[y_pred == 1, 0], X[y_pred == 1, 1], color="blue", label="Above the line"
    )
    plt.scatter(
        X[y_pred == -1, 0], X[y_pred == -1, 1], color="red", label="Below the line"
    )
    plt.scatter(
        X[y_pred == 0, 0],
        X[y_pred == 0, 1],
        color="green",
        marker="x",
        s=100,
        label="On the line",
    )

    x_range = np.linspace(min_value, max_value, n)
    plt.plot(x_range, 3 * x_range + 2, "k-", label="y = 3x + 2")
    plt.show()


if __name__ == "__main__":
    main()
