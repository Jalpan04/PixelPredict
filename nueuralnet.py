import numpy as np
import matplotlib.pyplot as plt

# ---------- Load MNIST ----------
def load_images(filepath):
    with open(filepath, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, num_rows * num_cols) / 255.0  # Normalize to [0, 1]

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# ---------- Activations ----------
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / expZ.sum(axis=0, keepdims=True)

# ---------- Loss and Accuracy ----------
def cross_entropy_loss(Y_pred, Y_true, epsilon=1e-12):
    m = Y_true.shape[0]
    correct_probs = Y_pred[Y_true, range(m)]
    log_likelihood = -np.log(correct_probs + epsilon)
    return np.sum(log_likelihood) / m

def accuracy(Y_pred, Y_true):
    preds = np.argmax(Y_pred, axis=0)
    return np.mean(preds == Y_true)

# ---------- Forward Pass ----------
def forward_pass(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# ---------- Backward Pass ----------
def backward_pass(X, Y_true, Z1, A1, Z2, A2, W2):
    m = X.shape[1]
    Y_one_hot = np.zeros_like(A2)
    Y_one_hot[Y_true, range(m)] = 1

    dZ2 = A2 - Y_one_hot
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (Z1 > 0).astype(float)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# ---------- Training ----------
def train(X_train, y_train, X_test, y_test, hidden_size=128, epochs=20, batch_size=64, learning_rate=0.01):
    input_size = 784
    output_size = 10

    # Initialize weights
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))

    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size].T
            y_batch = y_train_shuffled[i:i + batch_size]

            Z1, A1, Z2, A2 = forward_pass(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_pass(X_batch, y_batch, Z1, A1, Z2, A2, W2)

            # Gradient descent
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # Evaluate after each epoch
        _, _, _, A2_train = forward_pass(X_train.T, W1, b1, W2, b2)
        _, _, _, A2_test = forward_pass(X_test.T, W1, b1, W2, b2)

        train_loss = cross_entropy_loss(A2_train, y_train)
        test_loss = cross_entropy_loss(A2_test, y_test)
        train_acc = accuracy(A2_train, y_train)
        test_acc = accuracy(A2_test, y_test)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    return W1, b1, W2, b2

# ---------- Prediction ----------
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass(X, W1, b1, W2, b2)
    return A2

# ---------- Quick Visualization ----------
def plot_predictions(X, y, W1, b1, W2, b2, num_samples=10):
    indices = np.random.choice(X.shape[0], num_samples, replace=False)

    plt.figure(figsize=(num_samples * 2, 2))
    for i, idx in enumerate(indices):
        X_input = X[idx].reshape(784, 1)
        prediction = predict(X_input, W1, b1, W2, b2)
        predicted_label = np.argmax(prediction)
        image = X[idx].reshape(28, 28)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Act: {y[idx]}\nPred: {predicted_label}")
        plt.axis('off')

        print(f"Image {i + 1}: Actual label: {y[idx]}, Predicted: {predicted_label}")

    plt.tight_layout(pad=1.0)  # Adjust layout to prevent cutting off
    plt.show()


# ---------- Main ----------
if __name__ == "__main__":
    np.random.seed(42)

    # Load data
    X_train = load_images('data/train-images.idx3-ubyte')
    y_train = load_labels('data/train-labels.idx1-ubyte')
    X_test = load_images('data/t10k-images.idx3-ubyte')
    y_test = load_labels('data/t10k-labels.idx1-ubyte')

    # Train model
    W1, b1, W2, b2 = train(X_train, y_train, X_test, y_test)

    # Plot predictions
    plot_predictions(X_test, y_test, W1, b1, W2, b2, num_samples=10)
