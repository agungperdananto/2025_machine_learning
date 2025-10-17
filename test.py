input = [
[1, 1], [1, 0], [0, 1], [0, 0]
]
lr = [1, 1, 1, 1]
weights = [0, 0]
bias = 0
target =  [ 1, 1, 1, -1]

def net(x, w, b):
    xwi = 0
    for xi, wi in zip(x, w):
        xwi += xi * wi
    return xwi + b

def activation(n):
    if n > 0.2:
        return 1
    if n < -0.2:
        return -1
    return 0

for epoch in range(4):
    print(f"epoch {epoch}")
    for i, x in enumerate(input):
        n = net(x, weights, bias)
        y = activation(n)
        e = target[i] - y
        print(f"input: {x}, net: {n}, output: {y}, error: {e}")
        if e != 0:
            for j in range(len(weights)):
                weights[j] += lr[i] * e * x[j]
            bias += lr[i] * e
    print(f"weights: {weights}, bias: {bias}\n")