import numpy as np
import liteopt

target = np.random.rand(2)

l1 = 1.0
l2 = 1.0

def residual(x):
    r = np.zeros(2)
    px = l1 * np.cos(x[0]) + l2 * np.cos(x[0] + x[1])
    py = l1 * np.sin(x[0]) + l2 * np.sin(x[0] + x[1])
    r[0] = px - target[0]
    r[1] = py - target[1]
    return r

def jacobian(x):
    J = np.zeros((2, 2))
    s1 = np.sin(x[0])
    c1 = np.cos(x[0])
    s12 = np.sin(x[0] + x[1])
    c12 = np.cos(x[0] + x[1])

    J[0, 0] = -l1 * s1 - l2 * s12
    J[0, 1] = -l2 * s12
    J[1, 0] =  l1 * c1 + l2 * c12
    J[1, 1] =  l2 * c12
    return J

def main():
    x0 = [-1.2, 1.0]
    x_star, f_star, iters, rnorm, dxnorm, ok = liteopt.gn(
        residual,
        jacobian,
        x0=x0,
        verbose=True,
    )
    print("converged:", ok)
    print("x*:", x_star, "f(x*):", f_star)
    print("target:", target)
    print("iterations:", iters)
    print("final residual norm:", rnorm)
    print("final step norm:", dxnorm)

if __name__ == "__main__":
    main()