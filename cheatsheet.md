# Cheatsheet

## Arrays

~~~python
    A = np.array([1])
    A.shape

    B = np.arange(1,4, step=1) #vector will all numbers between 1 and 3 = [1 2 3]
~~~

### Matrices

~~~ python
    # -- creation matrices -- #
    np.array([[1,2,3], [4,5,6], [7,8,9]]) #3 x 3 matrix 
    np.matrix([[1, 2], [3, 4]]) #2 x 2 matrix
    np.matrix('1 2; 3 4') #2 x 2 matrix

    np.eye(3) #3 x 3 identity matrix

    np.zeros(2,3) #2 x 3 matrix of zeros

    np.ones(2,3) #2 x 3 matrix of ones

    np.random.rand(1000, 1000) #1000 x 1000 matrix with random values

    C = np.diag(B) #diagonal matrix from B
        # diag can be used to build a diagonal matrix from a vector or extract a diagonal from a matrix

    D = A + np.eye(3) #sum matrixes

    E = B.T #transpose matrix

    F = np.dot(A, E) #matrix product

    G = np.linalg.inv(D) #matrix inverse

    G1 = np.linalg.pinv(D) #matrix inverse with moore-penrose (//TODO pesquisar)

    # -- matrix concatenation -- #
    # When the axis to append is specified, the 
    # matrices/vectors must have the correct shape
    H = np.append([1, 2, 3], A)

    H1 = np.append(A, [[10, 11, 12]], axis = 0)
    H2 = np.append(A, [[4], [7], [10]], axis = 1)
    # print("Append [10, 11, 12] to A:")
    # print(H1)
    # print("Append [[4], [7], [10]] to A:")
    # print(H2)
~~~

### Matrixes operations

~~~python
    P = np.array([0, 0, 0, 0, 1],
    [0, 0, 0.5, 0.5, 0], 
    [0, 0.5, 0, 0, 0.5], 
    [0, 0.5, 0, 0, 0.5], 
    [0, 0, 1, 0, 0],
    )

    P5 = np.linalg.matrix_power(P,5) #power matrix to 5, must be a square matrix
~~~

### Indexing matrixes

~~~python
    A[0] #row 0
    A[1,2] #most efficient(not completely sure) to retrieve a value

    I = np.arange(10, 1, -1)
    #we can also index using a

    #1. a list
    print("I[[3, 3, 1, 8]]:", I[[3, 3, 1, 8]]) # >> [7 7 9 2]

    #2. a np.array
    print("I[np.array([3, 3, -3, 8])]:", I[np.array([3, 3, -3, 8])]) # >> [7 7 4 2]

    #3.2D np.array
    print("I[np.array([[1, 1], [2, 3]])]:", I[np.array([[1, 1], [2, 3]])]) # >> [[9 9]  [8 7]]


~~~

### Slicing matrixes

It is possible to retrieve/slice a part of a matrix.

~~~python
    # Rows between 1 and 2 (excluding the latter), 
    # columns between 0 and 1 (excluding the latter)
    print("A[1:2,0:1]:", A[1:2,0:1])

    # All rows except the last two,
    # every other column
    print("A[:-2,::2]:", A[:-2,::2])
~~~

## Plotting

trick: `%matplotlib inline` ipython draws the plots immediately after the cell

~~~python
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y_est)
    plt.plot(x, y, 'x') #displaying values as 'x'

    plt.xlabel('Input X')
    plt.ylabel('Output Y')

    plt.title('Linear regression')

    plt.subplot(nrows=1, ncols=2, 1) #this plot is the first plot
~~~

## Other

### Reading files

~~~ python
    data = np.load(filename)
~~~

### Timing

timing execution time:

~~~python
    import time

    t = time.time()
    #bla bla bla
    t1 = time.time() - t
~~~

### Bounds

Upperbound values numpy array `np.clip` <https://www.skytowner.com/explore/limiting_array_values_to_a_certain_range_in_numpy> (//TODO not sure if it the best way or if exists an efficient one)

### Random

Generate numbers:

- `rand`: generates random numbers **uniformly** from the interval $[0,1]$
- `randn`: generates **normaly** random numbers **uniformly** from the interval with *mean 0* and a *standard deviation of 1*.

Choose numbers:

Choose a value within a list of values with or without a probability associated `numpy.random.choice(a=<list of values>, ..., p=<probabilities>)` (lab0)
