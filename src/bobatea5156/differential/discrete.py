def diff(x, t):
    """
    Calculate the discrete derivative of x with respect to t.
    """

    if len(x) != len(t):
        raise ValueError("x and t must have the same length")

    # Calculate the discrete derivative using consecutive points.
    v = []

    for i in range(len(x) - 1):
        v.append((x[i + 1] - x[i]) / (t[i + 1] - t[i]))

    return v

