if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.utils as P

x = Variable(np.array([[1, 2, 3],[4, 5, 6]]))
y = x.transpose()
print(y)
y.backward()
print(x.grad)