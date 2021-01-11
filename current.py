if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
import unittest
from dezero import Variable, Function

# class Square(Function):
#     def forward(self, x):
#         y = np.power(x, 2)
#         return y

#     def backward(self, gy):
#         x = self.inputs[0].data
#         gx = 2 * x * gy
#         return gx

# def square(x):
#     return Square()(x)


# class Exp(Function):
#     def forward(self, x):
#         y = np.exp(x)
#         return y
    
#     def backward(self, gy):
#         x = self.inputs[0].data
#         gx = np.exp(x) * gy
#         return gx

# def exp(x):
#     return Exp()(x)

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2 * eps)

# class SquareTest(unittest.TestCase):
#     def test_forward(self):
#         x = Variable(np.array(2.0))
#         y = square(x)
#         expected = np.array(4.0)
#         self.assertEqual(y.data, expected)

#     def test_backward(self):
#         x = Variable(np.array(3.0))
#         y = square(x)
#         y.backward()
#         expected = np.array(6.0)
#         self.assertEqual(x.grad, expected)

#     def test_gradient_check(self):
#         x = Variable(np.random.rand(1))
#         y = square(x)
#         y.backward()
#         num_grad = numerical_diff(square, x)
#         flg = np.allclose(x.grad, num_grad)
#         self.assertTrue(flg)


if __name__ == '__main__':
    # unittest.main()

    x = Variable(np.array(np.pi/4))
    # x = Variable(np.array(0))
    y = my_sin(x)
    y.backward()

    print(y.data)
    print(x.grad)