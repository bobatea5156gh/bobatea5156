import secrets
import math


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""

    u = secrets.randbits(53) / (1 << 53)

    return a + (b - a) * u


def exponentialdist(lmbda: float) -> float:
    """Generate a cryptographically secure exponential sample."""

    y = uniform()

    while y == 0:
        y = uniform()

    return -math.log(y) / lmbda
def poissondist(lmbda: float) -> int:
    """Generate a Poisson-distributed random sample using inverse transform sampling."""

    u = uniform()

    k = 0
    probability = math.exp(-lmbda)
    cumulative = probability

    while cumulative < u:
        k += 1
        probability *= lmbda / k
        cumulative += probability

    return k