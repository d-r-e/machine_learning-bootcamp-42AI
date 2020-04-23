from mylinearregression import MyLinearRegression as MyLR
import numpy as np

if __name__ == "__main__":
    
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLR([[1.], [1.], [1.], [1.], [1]])

    print(mylr.predict_(X))
    # array([[8.], [48.], [323.]])

    print(mylr.cost_elem_(X,Y))
    # array([[37.5], [0.], [1837.5]])

    print(mylr.cost_(X,Y))
    # 1875.0

    mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=20000)
    print(mylr.theta)
    # array([[18.023..], [3.323..], [-0.711..], [1.605..], [-0.1113..]])

    print(mylr.predict_(X))
    # array([[23.499..], [47.385..], [218.079...]])

    print(mylr.cost_elem_(X,Y))
    # array([[0.041..], [0.062..], [0.001..]])

    print(mylr.cost_(X,Y))