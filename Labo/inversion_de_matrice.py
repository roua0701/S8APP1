import numpy as np
import matplotlib.pyplot as plt


def inversion_de_matrice_3():
    matrix_A = np.array([[3, 4, 1],
                  [5, 2, 3],
                  [6, 2, 2]], dtype=float)

    matrix_B = np.zeros((3, 3))

    matrix_I = np.eye(3)

    lr = 0.005
    epochs = 1000

    Ls = np.zeros((epochs,))

    for epoch in range(epochs):
        grad = 2 * (matrix_B @ matrix_A - matrix_I) @ matrix_A.T

        # MAJ des paramètres
        matrix_B -= lr * grad

        # Calcul du coût
        L = np.sum((matrix_B @ matrix_A - matrix_I) ** 2)
        Ls[epoch] = L

    print(matrix_B)
    print(matrix_B @ matrix_A)

    plt.plot(Ls)
    plt.xlabel("Epoch")
    plt.ylabel("Coût (L)")
    plt.title("Convergence de la descente de gradient")
    plt.show()


def inversion_de_matrice_6():
    matrix_A = np.array([[3, 4, 1, 2, 1, 5],
                  [5, 2, 3, 2, 2, 1],
                  [6, 2, 2, 6, 4, 5],
                  [1, 2, 1, 3, 1, 2],
                  [1, 5, 2, 3, 3, 3],
                  [1, 2, 2, 4, 2, 1]], dtype=float)

    n = matrix_A.shape[0]

    matrix_B = np.zeros((n, n))

    matrix_I = np.eye(n)

    lr = 0.003
    epochs = 100000

    Ls = np.zeros((epochs,))

    for epoch in range(epochs):
        grad = 2 * (matrix_B @ matrix_A - matrix_I) @ matrix_A.T

        # MAJ des paramètres
        matrix_B -= lr * grad

        # Calcul du coût
        L = np.sum((matrix_B @ matrix_A - matrix_I) ** 2)
        Ls[epoch] = L

    print(matrix_B)
    print(matrix_B @ matrix_A)

    plt.plot(Ls)
    plt.xlabel("Epoch")
    plt.ylabel("Coût (L)")
    plt.title("Convergence de la descente de gradient (6x6)")
    plt.show()

def inversion_de_matrice_4():
    matrix_A = np.array([[2, 1, 1, 2],
                  [1, 2, 3, 2],
                  [2, 1, 1, 2],
                  [3, 1, 4, 1]], dtype=float)

    n = matrix_A.shape[0]

    matrix_B = np.zeros((n, n))

    matrix_I = np.eye(n)

    lr = 0.001
    epochs = 1000

    Ls = np.zeros((epochs,))

    for epoch in range(epochs):
        grad = 2 * (matrix_B @ matrix_A - matrix_I) @ matrix_A.T

        # MAJ des paramètres
        matrix_B -= lr * grad

        # Calcul du coût
        L = np.sum((matrix_B @ matrix_A - matrix_I) ** 2)
        Ls[epoch] = L

    print(matrix_B)
    print(matrix_B @ matrix_A)

    plt.plot(Ls)
    plt.xlabel("Epoch")
    plt.ylabel("Coût (L)")
    plt.title("Convergence de la descente de gradient (4x4)")
    plt.show()


def main():
    # inversion_de_matrice_3()
    # inversion_de_matrice_6()
    inversion_de_matrice_4()

if __name__ == '__main__':
    main()