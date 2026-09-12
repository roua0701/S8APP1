import numpy as np
import matplotlib.pyplot as plt


def regression_polynomiale():
    x = np.array([-0.95, -0.82, -0.62, -0.43, -0.17, -0.07,
                   0.25, 0.38, 0.61, 0.79, 1.04])
    y = np.array([0.02, 0.03, -0.17, -0.12, -0.37, -0.25,
                  -0.10, 0.14, 0.53, 0.71, 1.53])

    N = 1  # ordre du polynôme

    matrix_a = np.zeros(N + 1)

    matrix_x = np.vstack([x**n for n in range(N + 1)]).T

    lr = 0.01
    epochs = 5000

    Ls = np.zeros((epochs,))

    for epoch in range(epochs):
        y_hat = matrix_a @ matrix_x.T

        grad = 2 * np.sum((y_hat - y) * matrix_x[epoch])

        matrix_a -= lr * grad

        L = np.sum((y_hat - y) ** 2)
        Ls[epoch] = L

    print("a =", matrix_a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(Ls)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Coût (L)")
    ax1.set_title("Convergence")
    ax1.set_yscale("log")

    x_plot = np.linspace(x.min(), x.max(), 200)
    X_plot = np.vstack([x_plot**n for n in range(N + 1)]).T
    y_plot = X_plot @ matrix_a

    ax2.scatter(x, y, label="Données")
    ax2.plot(x_plot, y_plot, color="red", label="Régression")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"Régression polynomiale (ordre {N})")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    regression_polynomiale()

if __name__ == '__main__':
    main()