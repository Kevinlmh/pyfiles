def bgd_optimizer(target_fn, grad_fn, init_W, X, Y, lr = 0.0001, tolerance = 1e-12, max_iter = 100000000):
    W = init_W
    target_value= target_fn(W, X, Y)
    for i in range(max_iter):
        grad=grad_fn(W, X, Y)
        next_W = W - grad * lr
        next_target_value = target_fn(next_W, X, Y)
        if abs(next_target_value - target_value) < tolerance:
            return i, next_W
        else : W, target_value = next_W, next_target_value
    return i, None