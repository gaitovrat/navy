import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        *,
        size=(2, 1, 1),
        learning_rate=0.1,
        epochs=1000,
        random_state: int | None = None,
    ):
        input_size, hidden_size, output_size = size

        if random_state:
            np.random.seed(random_state)

        # Weights for hidden layer
        self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))

        # Weights for output layer
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid activation function."""
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward propagation through the network."""
        # Hidden layer
        self.hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Output layer
        self.output_layer_input = (
            np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        )
        output = self.sigmoid(self.output_layer_input)

        return self.hidden_layer_output, output

    def backward(
        self, X: np.ndarray, y: np.ndarray, hidden_layer_output, output
    ) -> None:
        """Backpropagation to update weights and biases."""
        # Output layer error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Hidden layer error
        hidden_layer_error = np.dot(output_delta, self.weights_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(
            hidden_layer_output
        )

        # Update weights and biases
        self.weights_output += self.learning_rate * np.dot(
            hidden_layer_output.T, output_delta
        )
        self.bias_output += self.learning_rate * np.sum(
            output_delta, axis=0, keepdims=True
        )

        self.weights_hidden += self.learning_rate * np.dot(X.T, hidden_layer_delta)
        self.bias_hidden += self.learning_rate * np.sum(
            hidden_layer_delta, axis=0, keepdims=True
        )

    def train(self, X, y) -> None:
        """Train the neural network."""
        for _ in range(self.epochs):
            hidden_layer_output, output = self.forward(X)
            self.backward(X, y, hidden_layer_output, output)

    def predict(self, X) -> None:
        """Make predictions using the trained network."""
        _, output = self.forward(X)
        return np.round(output)


def main():
    data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    X = data[:, 0:2]
    y = data[:, 2].reshape(-1, 1)

    nn = NeuralNetwork(size=(2, 5, 1), random_state=42)
    nn.train(X, y)

    # Print results
    print("XOR Problem Results:")
    print("\nInput | Target | Prediction")
    print("-" * 30)
    for i in range(len(X)):
        prediction = nn.predict(X[i].reshape(1, -1))
        print(f"{X[i]} | {y[i][0]}      | {prediction[0][0]}")

    # Compute accuracy
    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
