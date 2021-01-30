if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.utils as P

x = Variable(np.array(1.0))
x.name = 'x'
y = F.sin(x)
y.name = 'y'
y.backward(create_graph=True)

for i in range(3):
    gx = x.grad
    x.grad.name = "gx" + str(i)
    x.cleargrad()
    gx.backward(create_graph=True)
    print(x.grad)

    if i == 2:
        P.plot_dot_graph(x.grad)