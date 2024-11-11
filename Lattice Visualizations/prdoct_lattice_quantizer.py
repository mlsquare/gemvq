import numpy as np

lookup = {(1, 1): 4}


def closest_center_encoding(v_i):
    return 1


def estimate(v, k=16):
    v_hat = []
    assert (len(v) % k) == 0
    for i in range(0, len(v), k):
        v_i = v[i: i + k]
        v_i_hat = closest_center_encoding(v_i)
        v_hat.append(v_i_hat)
    return v_hat


# Let's assume we compress both a and b
def estimate_product(a, b):
    a_hat = estimate(a)
    b_hat = estimate(b)
    prod = 0
    for i in range(len(a_hat)):
        prod += lookup[(a_hat[i], b_hat[i])]
    return prod


def main():
    vector1 = np.random.rand(48) * 2 - 1
    vector2 = np.random.rand(48) * 2 - 1
    print(estimate_product(vector1, vector2))


if __name__ == '__main__':
    main()


