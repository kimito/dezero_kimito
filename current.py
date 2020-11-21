if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from dezero import Variable

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

    def goldstein(x, y):
        z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2))  * \
            (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
        return z

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))

    z = goldstein(x, y)
    z.backward()

    print(x.grad, y.grad)