import numpy as np
import torch
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize model parameters
        self.w0 = torch.nn.Parameter(torch.tensor(0.0))
        self.w1 = torch.nn.Parameter(torch.tensor(0.0))

        # Optimizer and loss function
        self.optimizer = torch.optim.SGD(
            [self.w0, self.w1],
            lr=self.learning_rate
        )
        self.loss_fn = torch.nn.MSELoss()

        # Store training history
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

    def forward(self, X):
        """Compute y = w0 + w1*x."""
        X = torch.as_tensor(
            np.array(X, dtype=np.float32, copy=True),
            dtype=torch.float32
        ).reshape(-1)

        return self.w0 + self.w1 * X

    def fit(self, X_train, y_train, X_test, y_test):
        """Train the model and evaluate it on test data."""
        X_train = torch.as_tensor(
            np.array(X_train, dtype=np.float32, copy=True),
            dtype=torch.float32
        ).reshape(-1)

        y_train = torch.as_tensor(
            np.array(y_train, dtype=np.float32, copy=True),
            dtype=torch.float32
        ).reshape(-1)

        X_test = np.array(X_test, dtype=np.float32, copy=True).reshape(-1)
        y_test = np.array(y_test, dtype=np.float32, copy=True).reshape(-1)

        # Gradient descent
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            predictions = self.forward(X_train)
            loss = self.loss_fn(predictions, y_train)

            loss.backward()
            self.optimizer.step()

            # Save parameter and loss history
            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        # Evaluate the trained model on test data
        y_pred = self.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

        print(f"Test R-squared (R²): {r2}")

        return self

    def predict(self, X):
        """Predict AnnualProduction for new BCR values."""
        X = torch.as_tensor(
            np.array(X, dtype=np.float32, copy=True),
            dtype=torch.float32
        ).reshape(-1)

        with torch.no_grad():
            predictions = self.w0 + self.w1 * X

        return predictions.cpu().numpy()

    def analysis_plot(self, X, y):
        """Plot the data, fitted line, parameters, and training loss."""
        X = np.array(X, dtype=np.float32, copy=True).reshape(-1)
        y = np.array(y, dtype=np.float32, copy=True).reshape(-1)

        # Sort X so the regression line is drawn correctly
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        predictions = self.predict(X_sorted)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Original data and fitted regression line
        axes[0, 0].scatter(X, y, label="Original data")
        axes[0, 0].plot(
            X_sorted,
            predictions,
            label="Fitted regression line"
        )
        axes[0, 0].set_xlabel("Benefit-Cost-Ratio (BCR)")
        axes[0, 0].set_ylabel("AnnualProduction")
        axes[0, 0].set_title("Linear Regression")
        axes[0, 0].legend()

        # w0 history
        axes[0, 1].plot(self.w0_history)
        axes[0, 1].set_title("w0 during training")
        axes[0, 1].set_xlabel("Epoch")

        # w1 history
        axes[1, 0].plot(self.w1_history)
        axes[1, 0].set_title("w1 during training")
        axes[1, 0].set_xlabel("Epoch")

        # Loss history
        axes[1, 1].plot(self.loss_history)
        axes[1, 1].set_title("Training Loss")
        axes[1, 1].set_xlabel("Epoch")

        fig.tight_layout()
        fig.savefig("analysis_plot.png", dpi=150)
        plt.close(fig)

        print("Plots saved to analysis_plot.png")

        

# Cauchy Regression for Homework 04

import torch
import torch.nn as nn


class CauchyRegression(nn.Module):
    def __init__(self, n_features, c=1.0):
        super().__init__()

        if c <= 0:
            raise ValueError("c must be positive")

        self.linear = nn.Linear(n_features, 1)
        self.c = float(c)

    def forward(self, X):
        return self.linear(X)

    def cauchy_loss(self, y_pred, y_true):
        residual = y_true - y_pred

        loss = (self.c ** 2 / 2) * torch.log1p(
            (residual / self.c) ** 2
        )

        return torch.mean(loss)

    def fit(self, X, y, epochs=5000, lr=0.01):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr
        )

        self.loss_history = []
        self.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            y_pred = self.forward(X)
            loss = self.cauchy_loss(y_pred, y)

            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())

        return self

    def predict(self, X):
        self.eval()

        with torch.no_grad():
            return self.forward(X)

        
