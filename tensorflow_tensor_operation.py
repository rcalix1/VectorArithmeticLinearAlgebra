##########################################################################

## operations on tensors in tensorflow
## Ricardo A. Calix, Ph.D.

##########################################################################

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

##########################################################################

sess=tf.Session() #start a session

##########################################################################

 

#define tensors
a=tf.constant([[10,20],[30,40]]) #Dimension 2X2
b=tf.constant([5])
c=tf.constant([2,2])
d=tf.constant([[3],[3]])

#Run tensors to generate arrays
mat,scalar,one_d,two_d = sess.run([a,b,c,d])

print("mat")
print(mat)

print("scalar")
print(scalar)

print("one_d")
print(one_d)

print("two_d")
print(two_d)


#broadcast multiplication with scalar
print( "  sess.run(tf.multiply(mat,scalar))  "  )
print(   sess.run(tf.multiply(mat,scalar))    )

#broadcast multiplication with 1_D array (Dimension 1X2)
print( "  sess.run(tf.multiply(mat,one_d))   "  )
print(   sess.run(tf.multiply(mat,one_d))     )

#broadcast multiply 2_d array (Dimension 2X1)
print( "   sess.run(tf.multiply(mat,two_d))    "   )
print(    sess.run(tf.multiply(mat,two_d))       )

'''

    mat
      [[10 20]
       [30 40]]
       
      scalar
      [5]
      
      one_d
      [2 2]
      
      two_d
      [[3]
       [3]]
       
        sess.run(tf.multiply(mat,scalar))
      [[ 50 100]
       [150 200]]
       
        sess.run(tf.multiply(mat,one_d))
      [[20 40]
       [60 80]]
       
         sess.run(tf.multiply(mat,two_d))
      [[ 30  60]
       [ 90 120]]


'''


print("************************************************")

#####################################################################

arr = np.array([1, 5.5, 32, 11, 20])

tensor1 = tf.convert_to_tensor(arr,tf.float64)

print(tensor1)

'''

 Tensor("Const_4:0", shape=(5,), dtype=float64)

'''

print("************************************************")

#####################################################################

arr = np.array([1, 5.5, 3, 11, 30])

tensor = tf.convert_to_tensor(arr,tf.float64)

print(sess.run(tensor))

print(sess.run(tensor[1]))

'''

    [ 1.   5.5  3.  11.  30. ]
    
       5.5

'''

print("************************************************")

#####################################################################

d2arr = np.array([ [1, 5.5, 3, 15, 20],
                   [10, 22, 30, 4, 50],
                   [60, 70, 83, 90, 101] ])
                  

tensor = tf.convert_to_tensor(d2arr)

print(  sess.run(tensor)  )

'''

    [[  1.    5.5   3.   15.   20. ]
     [ 10.   22.   30.    4.   50. ]
     [ 60.   70.   83.   90.  101. ]]

'''

print("************************************************")

#####################################################################
'''
tf.multiply(a,b) is identical to a*b
f.multiply(X, Y) does element-wise multiplication so that

    [[1 2]    [[1 3]      [[1 6]
     [3 4]] .  [2 1]]  =   [6 4]]
    
wheras tf.matmul does matrix multiplication so that

    [[1 0]    [[1 3]      [[1 3]
     [0 1]] .  [2 1]]  =   [2 1]]
    
using tf.matmul(X, X, transpose_b=True) means that you are calculating X . X^T where ^T
indicates the transposing of thmatrix and . is the matrix multiplication.


'''

a = tf.constant([[1, 2],
                 [3, 4]])
                 
b = tf.constant([[1, 1],
                 [1, 1]])

print(sess.run(   tf.add(a, b)         ))
print(sess.run(   tf.multiply(a, b)    ))
print(sess.run(   tf.matmul(a, b)      ))

'''

    [[2 3]
     [4 5]]
     
    [[1 2]
     [3 4]]
     
    [[3 3]
     [7 7]]

'''


print("************************************************")

#####################################################################

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

print( sess.run(    c    )   )

# Find the largest value
print(" sess.run(     tf.reduce_max(c)    )  " )
print( sess.run(     tf.reduce_max(c)    )   )

# Find the index of the largest value
print( "  sess.run(    tf.argmax(c)   ) "  )
print(   sess.run(    tf.argmax(c)   )   )

# Compute the softmax
print(  "  sess.run(    tf.nn.softmax(c)    )  " )
print(    sess.run(    tf.nn.softmax(c)    )   )


'''

     [[ 4.  5.]
      [10.  1.]]
      
      sess.run(     tf.reduce_max(c)    )
     10.0
     
     
       sess.run(    tf.argmax(c)   )
     [1 0]
     
     
       sess.run(    tf.nn.softmax(c)    )
     [[2.6894143e-01 7.3105860e-01]
      [9.9987662e-01 1.2339458e-04]]

'''

print("************************************************")

#####################################################################

rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("Type of every element:", rank_4_tensor.dtype)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])

'''

    
    Type of every element: <dtype: 'float32'>
    Shape of tensor: (3, 2, 4, 5)
    Elements along axis 0 of tensor: 3
    Elements along the last axis of tensor: 5

'''

print("************************************************")

#####################################################################
## Broadcasting

x = tf.constant([1, 2, 3])
y = tf.constant(2)
z = tf.constant([2, 2, 2])

print(sess.run(x))
print(sess.run(y))
print(sess.run(z))
# All of these are the same computation
print("sess.run(tf.multiply(x, 2))")
print(sess.run(tf.multiply(x, 2)))

print("sess.run(x * y)")
print(sess.run(x * y))

print("sess.run(x * z)")
print(sess.run(x * z))


'''

    
    [1 2 3]
    
    2
    
    [2 2 2]
    
    sess.run(tf.multiply(x, 2))
    [2 4 6]
    
    
    sess.run(x * y)
    [2 4 6]
    
    
    sess.run(x * z)
    [2 4 6]


'''

print("************************************************")

#####################################################################
## Broadcasting
## In this case a 3x1 matrix is element-wise multiplied by a 1x4 matrix to
## produce a 3x4 matrix. Note how the leading 1 is optional:
## The shape of y is [4].

x = tf.constant([1, 2, 3])
y = tf.constant(2)
z = tf.constant([2, 2, 2])

print(sess.run(x))
print(sess.run(y))
print(sess.run(z))


x = tf.reshape(x,[3,1])
y = tf.range(1, 5)


print(x, "\n")
print(sess.run(x))

print(y, "\n")
print(sess.run(y))

print(tf.multiply(x, y))
print(sess.run(tf.multiply(x, y)))

'''

    [1 2 3]
    
     2
     
     [2 2 2]
     
     Tensor("Reshape:0", shape=(3, 1), dtype=int32)

     [[1]
      [2]
      [3]]
      
      
     Tensor("range:0", shape=(4,), dtype=int32)

     [1 2 3 4]
     
     
     
     Tensor("Mul_7:0", shape=(3, 4), dtype=int32)
     [[ 1  2  3  4]
      [ 2  4  6  8]
      [ 3  6  9 12]]

'''


print("************************************************")



#####################################################################

sess.close()

#####################################################################

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")

