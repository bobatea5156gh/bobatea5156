import torch


def rowswap(A, source_row, target_row):
    """Swap two rows of a matrix."""
    A = A.clone()
    A[[source_row, target_row]] = A[[target_row, source_row]]
    return A


def rowscale(A, source_row, scaling_factor):
    """Scale a row by a given factor."""
    A = A.clone()
    A[source_row] = scaling_factor * A[source_row]
    return A


def rowreplacement(A, first_row, second_row, j, k):
    """Perform j*R_first_row + k*R_second_row."""
    A = A.clone()
    A[second_row] = j * A[first_row] + k * A[second_row]
    return A


def rref(A):
    """Return the reduced row echelon form of a matrix."""
    A = A.clone().to(torch.float64)

    rows, cols = A.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Find a suitable pivot row.
        pivot = None
        for row in range(pivot_row, rows):
            if torch.abs(A[row, col]) > 1e-10:
                pivot = row
                break

        # No pivot in this column.
        if pivot is None:
            continue

        # Move the pivot row into position.
        if pivot != pivot_row:
            A = rowswap(A, pivot, pivot_row)

        # Make the pivot equal to 1.
        pivot_value = A[pivot_row, col]
        A = rowscale(A, pivot_row, 1.0 / pivot_value)

        # Make all other entries in this column zero.
        for row in range(rows):
            if row != pivot_row:
                factor = A[row, col]
                if torch.abs(factor) > 1e-10:
                    A = rowreplacement(
                        A,
                        pivot_row,
                        row,
                        -factor,
                        1.0
                    )

        pivot_row += 1

    # Remove very small numerical values.
    A[torch.abs(A) < 1e-10] = 0

    return A


if __name__ == "__main__":
    A = torch.tensor([
        [1.0, 3.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 1.0, -4.0]
    ])

    print("Original matrix:")
    print(A)

    A1 = rowswap(A, 0, 1)
    print("\nAfter R1 <-> R2:")
    print(A1)

    A2 = rowscale(A1, 0, 1 / 3)
    print("\nAfter (1/3)R1:")
    print(A2)

    A3 = rowreplacement(A2, 0, 1, -3, 1)
    print("\nAfter R2 = -3R1 + R2:")
    print(A3)

    print("\nRREF:")
    print(rref(A))