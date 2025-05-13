def step_euler(f, x, u_c, u_e, w, delta_t):
    x_next = x + delta_t * f(x, u_c, u_e, w)
    return x_next


def step_rk4(f, x, u_c, u_e, w, delta_t):
    k1 = f(x, u_c, u_e, w)
    k2 = f(x + delta_t / 2 * k1, u_c, u_e, w)
    k3 = f(x + delta_t / 2 * k2, u_c, u_e, w)
    k4 = f(x + delta_t * k3, u_c, u_e, w)
    x_next = x + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next
