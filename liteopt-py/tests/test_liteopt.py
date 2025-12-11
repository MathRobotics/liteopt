import numpy as np
import liteopt

def f(x):
    x = np.asarray(x)
    return float((1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2)

def grad(x):
    x = np.asarray(x)
    df_dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
    df_dy = 200.0 * (x[1] - x[0]**2)
    return [float(df_dx), float(df_dy)]

def main():
    x0 = [-1.2, 1.0]
    x_star, f_star, converged = liteopt.gd(
        f, grad, x0,
        step_size=1e-3,
        max_iters=200_000,
        tol_grad=1e-10,
    )
    print("converged:", converged)
    print("x*:", x_star, "f(x*):", f_star)
    print("expected x*:", [1.0, 1.0], "f(x*):", f([1.0, 1.0]))

if __name__ == "__main__":
    main()
