#########################################################
## Numpy examples and techniques
## Ricardo A. Calix, Ph.D.
#########################################################

import numpy as np
import matplotlib.pyplot as plt


  
########################################################

a = np.array([4,5,2,6,8])
print(a)

'''

    [4 5 2 6 8]
    

'''


print("****************************")

########################################################

a = np.array([1, 3, 2, 5] , dtype='float32'   )
print(a)

'''


    [1. 3. 2. 5.]
    

'''

print("****************************")

########################################################

list_of_lists = [[1, 2, 3], [4, 4, 5] , [6, 2]]
b = np.array(list_of_lists)
print(b)

'''

  
    [list([1, 2, 3]) list([4, 4, 5]) list([6, 2])]
   

'''


print("****************************")

########################################################

list_of_lists = [[1, 2, 3], [4, 4, 5] , [6, 2, 11]]
b = np.array(list_of_lists)
print(b)

'''

  
       [[ 1  2  3]
        [ 4  4  5]
        [ 6  2 11]]
  

'''


print("****************************")

########################################################


b = np.zeros(10, dtype=int)
print(b)

'''

      [0 0 0 0 0 0 0 0 0 0]

'''


print("****************************")

#########################################################

b = np.ones((4, 6), dtype=float)
print(b)

'''


    [[1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]]
   

'''



print("****************************")

#########################################################

b = np.full((3, 3), 42)
print(b)

'''

       [[42 42 42]
        [42 42 42]
        [42 42 42]]
       

'''


print("****************************")


#########################################################

b = np.arange(1, 30, 3)
print(b)

'''

    [ 1  4  7 10 13 16 19 22 25 28]
    

'''


print("****************************")

#########################################################

b = np.linspace(0, 1, 20)
print(b)

'''

    
    [0.         0.05263158 0.10526316 0.15789474 0.21052632 0.26315789
     0.31578947 0.36842105 0.42105263 0.47368421 0.52631579 0.57894737
     0.63157895 0.68421053 0.73684211 0.78947368 0.84210526 0.89473684
     0.94736842 1.        ]
    


'''
print("****************************")


#########################################################

b = np.random.random((4, 4))
print(b)

'''

   
    [[0.52467069 0.68216617 0.79782109 0.33720887]
     [0.67956722 0.04082517 0.31311017 0.72985649]
     [0.64533659 0.83448976 0.37986602 0.60524177]
     [0.98868748 0.36999339 0.33000013 0.04157917]]
    
'''

print("****************************")

#########################################################
## mean 0 and standard deviation 1


b = np.random.normal(0, 1, (4,4))
print(b)

'''

   
    [[ 0.82064949 -0.95219825 -1.27123377 -1.01187383]
     [ 0.44419588  0.17695603 -0.75775624 -0.14476445]
     [ 0.59233303  1.27530445  0.77260354 -0.80240966]
     [-0.58009786 -1.04106833 -1.27650071  0.28198804]]
   

'''

print("****************************")

#########################################################
## indentity matrix

b = np.eye(5)
print(b)

'''

   
       [[1. 0. 0. 0. 0.]
        [0. 1. 0. 0. 0.]
        [0. 0. 1. 0. 0.]
        [0. 0. 0. 1. 0.]
        [0. 0. 0. 0. 1.]]
      

'''

print("****************************")

#########################################################

b1 = np.random.randint(20, size=6)
b2 = np.random.randint(20, size=(3,4))
b3 = np.random.randint(20, size=(2,4,6))
print(b2)
print(b3)
print("b2 dims ", b2.ndim)
print("b3 shape ", b3.shape)
print("b2 size ", b2.size)
print("data type of b3 ", b3.dtype)

'''

    
          [[ 7 13 16 13]
           [11  6 17 11]
           [ 0 12  6  4]]
           
          [[[12 10 15  3 14  4]
            [ 9  7  4  0 16 10]
            [11 16  9  0  5 12]
            [ 4 12  6  9  3  5]]
            

           [[ 7 17  5 18  0 15]
            [ 9  3  4  4  7  0]
            [15  1  4 12 10 17]
            [ 9 14  1 14 19  8]]]
            
          b2 dims  2
          b3 shape  (2, 4, 6)
          b2 size  12
          data type of b3  int64
          

'''

print("****************************")

#########################################################
## indexing

a = np.array([1, 3, 2, 5] , dtype='float32'   )
print(a)
print("first ", a[0])
print("third ", a[2])
print("last ", a[-1])
print("before last ", a[-2])

'''

   
    [1. 3. 2. 5.]
    first  1.0
    third  2.0
    last  5.0
    before last  2.0
    



'''

print("****************************")

#########################################################
## indexing

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )
print(a)
print("first ", a[0,0])

print("last ", a[2, -1])


'''

    
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
    first  1
    last  12
   

'''

print("****************************")


########################################################
## slicing

x = np.arange(15)
print(x)
print("first 4 elemets ", x[:4])
print("all after 3 ", x[3:])
print("even indeces ", x[::2] )    ## starts at 0     ## 2 is the step size
print("uneven indeces ", x[1::2])    ## starts at 1
print("reverse ", x[::-1]  )     ## step value is negative starts at last element

'''

    
       [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
       first 4 elemets  [0 1 2 3]
       all after 3  [ 3  4  5  6  7  8  9 10 11 12 13 14]
       even indeces  [ 0  2  4  6  8 10 12 14]
       uneven indeces  [ 1  3  5  7  9 11 13]
       reverse  [14 13 12 11 10  9  8  7  6  5  4  3  2  1  0]
       


'''

print("****************************")

########################################################
## slicing

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )
print(a)
print(a[:2,:2])

'''

   
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
     
    [[1 2]
     [5 6]]
     
    

'''

print("****************************")

########################################################
## 

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )
print(a)
print("a[:-1,:-1]")
print(a[:-1,:-1])

'''

   
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
     
    a[:-1,:-1]
    
    [[1 2 3]
     [5 6 7]]
     
    

'''

print("****************************")


########################################################
##

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )
print(a)
print("a[:-1,:-2]")
print(a[:-1,:-2])

'''

   
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
     
    a[:-1,:-2]
    
    [[1 2]
     [5 6]]
   

'''

print("****************************")


########################################################
## shifting

x = np.arange(15)
print(x)
print("shift right  ", x[:-1]   )
print("shift left   ",  x[1:]   )


'''

  
                     [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
                     
       shift right   [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13]
       shift left    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14]
       

'''

print("****************************")


########################################################

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )
print(a)
print("column 1 ")
print(a[:, 1])

'''

    
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
     
    column 1
    
    [ 2  6 10]
   

'''

print("****************************")

########################################################

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )
print(a)
print("row 1 ")
print(a[1, :])

'''


       [[ 1  2  3  4]
        [ 5  6  7  8]
        [ 9 10 11 12]]
        
       row 1
       
       [5 6 7 8]
      

'''

print("****************************")


########################################################
## slicing in numpy does not copy array but
## instead still modifies the original.
## to copy  use .copy()


a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )
new_a = a[:2, :2]
print(a)
print(new_a)
new_a[0,0] = 42
print(a)
new_a2 = a[:2, :2].copy()
new_a2[0,0] = 17
print(a)

'''

   
          [[ 1  2  3  4]
           [ 5  6  7  8]
           [ 9 10 11 12]]
           
          [[1 2]
           [5 6]]
           
          [[42  2  3  4]
           [ 5  6  7  8]
           [ 9 10 11 12]]
           
          [[42  2  3  4]
           [ 5  6  7  8]
           [ 9 10 11 12]]
           
        


'''

print("****************************")

#########################################################

a = np.arange(1, 10)
b = a.reshape( (3,3)   )
print(a)
print(b)

'''

 
            [1 2 3 4 5 6 7 8 9]
            
            [[1 2 3]
             [4 5 6]
             [7 8 9]]
             
           

'''

print("****************************")

#########################################################

v = np.array(  [1, 2, 3, 4, 5]  )
v1 = np.array(  [5, 5, 5, 5, 5]  )

m = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )

print("reshape as row vector with reshape ", v.reshape(  (1,5)  ))
print("reshape as row vector with newaxis ",  v1[np.newaxis, :]   )
print("reshape as column vector with newaxis ")
print( v1[:, np.newaxis]   )
print(" reshape matrix  m[:, np.newaxis, np.newaxis, :] with newaxis ")
print(  m[:, np.newaxis, np.newaxis, :]  )

'''
    [
        [
           [
              [ 1  2  3  4]
           ]
        ]

        [
           [
              [ 5  6  7  8]
           ]
        ]

        [
           [
              [ 9 10 11 12]
           ]
        ]
    ]
    
    
    
    
               reshape as row vector with reshape  [[1 2 3 4 5]]
               reshape as row vector with newaxis  [[5 5 5 5 5]]
               
               reshape as column vector with newaxis
               [[5]
                [5]
                [5]
                [5]
                [5]]
                
                reshape matrix  m[:, np.newaxis, np.newaxis, :] with newaxis
               [[[[ 1  2  3  4]]]

                [[[ 5  6  7  8]]]

                [[[ 9 10 11 12]]]]
              
    
    

'''




print("****************************")

#########################################################

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
c = np.array([9,10,11])
ab = np.concatenate(  [a, b]  )
abc = np.concatenate(  [a, b, c]  )
print(a)
print(b)
print(c)
print("concatenate a with b ", ab)
print("concatenate a with b with c", abc)

'''


                  [1 2 3 4]
                  [5 6 7 8]
                  [ 9 10 11]
                  
                  concatenate a with b  [1 2 3 4 5 6 7 8]
                  concatenate a with b with c [ 1  2  3  4  5  6  7  8  9 10 11]
                
       
    
'''

print("****************************")


#########################################################

m1 = np.array([ [1,2,3],
                [4,5,6],
                [7,8,9]  ])
                
m2 = np.array([ [10,11,12],
                [13,14,14],
                [16,17,18]  ])
                
print(m1)
print(m2)
m1_m2_concat = np.concatenate([m1, m2], axis=0)
print("m1 m2 concat axis 0 ")
print(m1_m2_concat)


'''

    
                    [[1 2 3]
                     [4 5 6]
                     [7 8 9]]
                     
                    [[10 11 12]
                     [13 14 14]
                     [16 17 18]]
                     
                    m1 m2 concat axis 0
                    [[ 1  2  3]
                     [ 4  5  6]
                     [ 7  8  9]
                     [10 11 12]
                     [13 14 14]
                     [16 17 18]]
                   
         

'''

print("****************************")

#########################################################

m1 = np.array([ [1,2,3],
                [4,5,6],
                [7,8,9]  ])
                
m2 = np.array([ [10,11,12],
                [13,14,14],
                [16,17,18]  ])
                
print(m1)
print(m2)
m1_m2_concat = np.concatenate([m1, m2], axis=1)
print("m1 m2 concat axis 1 ")
print(m1_m2_concat)


'''

  
                       [[1 2 3]
                        [4 5 6]
                        [7 8 9]]
                        
                       [[10 11 12]
                        [13 14 14]
                        [16 17 18]]
                        
                       m1 m2 concat axis 1
                       [[ 1  2  3 10 11 12]
                        [ 4  5  6 13 14 14]
                        [ 7  8  9 16 17 18]]
                        
                      

'''

print("****************************")

#########################################################
## np.vstack   -->>    vertical stack
## np.hstack   -->>    horizontal stack

m1 = np.array([ [1,2,3],
                [4,5,6],
                [7,8,9]  ])
                
m2 = np.array([ [10,11,12],
                [13,14,14],
                [16,17,18]  ])
                
print(m1)
print(m2)
m1_m2_concat = np.vstack( [m1, m2]  )
print(" vstack ")
print(m1_m2_concat)

m1_m2_concat2 = np.hstack( [m1, m2]  )
print(" hstack ")
print(m1_m2_concat2)

'''

 
                          [[1 2 3]
                           [4 5 6]
                           [7 8 9]]
                           
                          [[10 11 12]
                           [13 14 14]
                           [16 17 18]]
                           
                           vstack
                          [[ 1  2  3]
                           [ 4  5  6]
                           [ 7  8  9]
                           [10 11 12]
                           [13 14 14]
                           [16 17 18]]
                           
                           hstack
                          [[ 1  2  3 10 11 12]
                           [ 4  5  6 13 14 14]
                           [ 7  8  9 16 17 18]]
                      


'''

print("****************************")

#########################################################
## math opeations in numpy

x = np.array([1,2,3,4])
print(x)
print("x + 10 ", x+10)
print("x - 10 ", x-10)
print("x * 10 ", x*10)
print("x / 2 ", x/2)
print("  -x ", -x)
print("x ** 3", x ** 3)
print("4^x", np.power(4, x) )
print("  np.log(x) ", np.log(x))
print(" np.log2(x)", np.log2(x))
print("np.log10(x)", np.log10(x) )

'''


            [1 2 3 4]
            x + 10  [11 12 13 14]
            x - 10  [-9 -8 -7 -6]
            x * 10  [10 20 30 40]
            x / 2  [0.5 1.  1.5 2. ]
            -x  [-1 -2 -3 -4]
            x ** 3 [ 1  8 27 64]
            4^x [  4  16  64 256]
            np.log(x)  [0.         0.69314718 1.09861229 1.38629436]
            np.log2(x) [0.        1.        1.5849625 2.       ]
            np.log10(x) [0.         0.30103    0.47712125 0.60205999]
                        

'''




print("****************************")

#########################################################

               ##    min   max    n_samples
angles = np.linspace(0,     4,    10)
print(angles)
print("np.sin")
print(np.sin(angles))

'''

  
[0.         0.44444444 0.88888889 1.33333333 1.77777778 2.22222222
                2.66666667 3.11111111 3.55555556 4.        ]
                             
np.sin
[ 0.          0.42995636  0.77637192  0.9719379   0.9786557   0.79522006
            0.45727263  0.03047682 -0.40224065 -0.7568025 ]
                           


'''


print("****************************")


#########################################################
## aggregates and others
## tf.reduce_sum()  in tensorflow

x = np.array( [1 , 2, 3, 4, 5]  )
print(x)
print("  np.add.reduce(x)  = ",  np.add.reduce(x)  )
print("  np.multiply.reduce(x)  = ",  np.multiply.reduce(x)  )
print("  np.sum(x)  = ",  np.sum(x)  )
print("  np.min(x)  = ",  np.min(x)  )
print("  np.max(x)  = ",  np.max(x)  )

'''

 
            [1 2 3 4 5]
            np.add.reduce(x)  =  15
            np.multiply.reduce(x)  =  120
            np.sum(x)  =  15
            np.min(x)  =  1
            np.max(x)  =  5
                              


'''

print("****************************")

#########################################################


m = np.array( [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16] ])

print(m)

print("  m.sum()   = ", m.sum()  )
print("  np.min(m, axis=0)  = ", np.min(m, axis=0) )
print("  np.min(m, axis=1)  = ", np.min(m, axis=1) )
print("  np.min(m, axis=-1)  (-1 is last item) = ", np.min(m, axis=-1) )

'''

   
                                  [[ 1  2  3  4]
                                   [ 5  6  7  8]
                                   [ 9 10 11 12]
                                   [13 14 15 16]]
                m.sum()   =  136
                np.min(m, axis=0)  =  [1 2 3 4]
                np.min(m, axis=1)  =  [ 1  5  9 13]
                np.min(m, axis=-1)  (-1 is last item) =  [ 1  5  9 13]
                                  


'''



print("****************************")

#########################################################
## broadcasting

a = np.array(  [0, 1, 2]   )
m = np.ones(   (3,3)    )
ma = m + a
print(" m ")
print(m)
print(" a ")
print(a)
print("m + a ")
print(ma)
a = a[:, np.newaxis]
print("a[:, np.newaxis]")
print(a)
print("m + a ")
ma = m + a
print(ma)

'''

   
     m
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
     
     a
    [0 1 2]
    
    m + a
    [[1. 2. 3.]
     [1. 2. 3.]
     [1. 2. 3.]]
     
    a[:, np.newaxis]
    [[0]
     [1]
     [2]]
     
    m + a
    [[1. 1. 1.]
     [2. 2. 2.]
     [3. 3. 3.]]
    

'''


print("****************************")

#########################################################
## broadcasting
## 2 vectors multiplied to get a matrix

v1 = np.array(  [1, 1, 1]   )
v2 = np.array(  [0, 1, 2]   )[:, np.newaxis]

print(v1)
print(v2)

vvplus = v1 + v2
v1v2   = v1 * v2

print("v1 + v2")
print(v1 + v2)
print("v1 * v2")
print(v1 * v2)

'''

  
    [1 1 1]
    [[0]
     [1]
     [2]]
     
    v1 + v2
    [[1 1 1]
     [2 2 2]
     [3 3 3]]
     
    v1 * v2
    [[0 0 0]
     [1 1 1]
     [2 2 2]]
   

'''


print("****************************")

#########################################################
## broadcasting
## 2 vectors multiplied to get a matrix

v1 = np.array(  [1, 1, 1]   )[:, np.newaxis]
v2 = np.array(  [0, 1, 2]   )

print(v1)
print(v2)

vvplus = v1 + v2
v1v2   = v1 * v2

print("v1 + v2")
print(v1 + v2)
print("v1 * v2")
print(v1 * v2)

'''


       [[1]
        [1]
        [1]]
        
       [0 1 2]
       
       v1 + v2
       [[1 2 3]
        [1 2 3]
        [1 2 3]]
        
       v1 * v2
       [[0 1 2]
        [0 1 2]
        [0 1 2]]
    

'''

print("****************************")


#########################################################
## masks
## boolean

m = np.array( [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16] ])

print(m)


print("m < 5")
print(m < 5)

print("m > 11")
print(m > 11)

print("m == 14")
print(m == 14)

print("np.equal(m, 13)")
print(np.equal(m, 13))

print("np.equal(m, 13)")
print(np.equal(m, 13))

print("np.sum(m < 12, axis=1)")
print("how many values less than 12 in each row?")
print(np.sum(m < 12, axis=1)[:, np.newaxis])

print("m[m < 8]")
print(m[m < 8])

'''


        [[ 1  2  3  4]
         [ 5  6  7  8]
         [ 9 10 11 12]
         [13 14 15 16]]
         
        m < 5
        [[ True  True  True  True]
         [False False False False]
         [False False False False]
         [False False False False]]
         
        m > 11
        [[False False False False]
         [False False False False]
         [False False False  True]
         [ True  True  True  True]]
         
        m == 14
        [[False False False False]
         [False False False False]
         [False False False False]
         [False  True False False]]
         
        np.equal(m, 13)
        [[False False False False]
         [False False False False]
         [False False False False]
         [ True False False False]]
         
        np.equal(m, 13)
        [[False False False False]
         [False False False False]
         [False False False False]
         [ True False False False]]
         
        np.sum(m < 12, axis=1)
        how many values less than 12 in each row?
        [[4]
         [4]
         [3]
         [0]]
         
        m[m < 8]
        [1 2 3 4 5 6 7]
      


'''

print("****************************")


#########################################################
## masks
## boolean
## fancy indexin or slicing

m = np.array( [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16] ])

print(m)
row = np.array(  [0, 1, 2] )
col = np.array(  [0, 1, 2] )
print("row = np.array(  [0, 1, 2] )")
print("col = np.array(  [0, 1, 2] )")
print(" m[row,col] ")
print(   m[row,col]   )
print("m[2:, [0, 2]] ")
print( m[2:, [0, 2]]  )


'''


          [[ 1  2  3  4]
           [ 5  6  7  8]
           [ 9 10 11 12]
           [13 14 15 16]]
           
          row = np.array(  [0, 1, 2] )
          col = np.array(  [0, 1, 2] )
          
           m[row,col]
          [ 1  6 11]
          
          m[2:, [0, 2]]
          [[ 9 11]
           [13 15]]




'''

print("****************************")

#########################################################


m = np.array( [[1, 2],
               [3, 4],
               [5, 6],
               [7, 8],
               [9, 10],
               [11, 12],
               [13, 14],
               [15, 16],
               [17, 18],
               [19, 20] ])
               
print(m)
m1 = m[:, np.newaxis, :]  ## for broadcasting
print(m1.shape)
print(m1)

m2 = m[np.newaxis, :,:]  ## for broadcasting
print(m2.shape)
print(m2)

diff = (m1 - m2)**2
print(diff.shape)
print("  diff = (m1 - m2)**2 ")
print(diff)
print("np.sum(diff, axis=-1)")
print(np.sum(diff, axis=-1))

'''

    
    [[ 1  2]
     [ 3  4]
     [ 5  6]
     [ 7  8]
     [ 9 10]
     [11 12]
     [13 14]
     [15 16]
     [17 18]
     [19 20]]
     
    (10, 1, 2)
    
    
    
    [[[ 1  2]]

     [[ 3  4]]

     [[ 5  6]]

     [[ 7  8]]

     [[ 9 10]]

     [[11 12]]

     [[13 14]]

     [[15 16]]

     [[17 18]]

     [[19 20]]]
     
     
    (1, 10, 2)
    
    
    
    [[[ 1  2]
      [ 3  4]
      [ 5  6]
      [ 7  8]
      [ 9 10]
      [11 12]
      [13 14]
      [15 16]
      [17 18]
      [19 20]]]
      
      
    (10, 10, 2)
    
    
      diff = (m1 - m2)**2
    [[[  0   0]
      [  4   4]
      [ 16  16]
      [ 36  36]
      [ 64  64]
      [100 100]
      [144 144]
      [196 196]
      [256 256]
      [324 324]]

    ...
    
     [[324 324]
      [256 256]
      [196 196]
      [144 144]
      [100 100]
      [ 64  64]
      [ 36  36]
      [ 16  16]
      [  4   4]
      [  0   0]]]
      
      
      
    np.sum(diff, axis=-1)
    [[  0   8  32  72 128 200 288 392 512 648]
     [  8   0   8  32  72 128 200 288 392 512]
     [ 32   8   0   8  32  72 128 200 288 392]
     [ 72  32   8   0   8  32  72 128 200 288]
     [128  72  32   8   0   8  32  72 128 200]
     [200 128  72  32   8   0   8  32  72 128]
     [288 200 128  72  32   8   0   8  32  72]
     [392 288 200 128  72  32   8   0   8  32]
     [512 392 288 200 128  72  32   8   0   8]
     [648 512 392 288 200 128  72  32   8   0]]


'''

print("****************************")





#########################################################

##  [64, 40, 512]
batch = np.random.normal(0, 1,  (64, 40, 512))
print(batch)


## np.newaxis changes i from [512,]   to [1, 512]
i   = np.arange(512)[np.newaxis, :]

print("i")
print(i)


## np.newaxis changes pos from [40,]   to [40, 1]
pos = np.arange(40)[:, np.newaxis]

print("pos")
print(pos)


angle_rates = 1 / np.power(10000,     (2 * (i // 2)) / 512.0      )

print(" angle_rates = 1 / np.power(10000, (2 * (i // 2)) / 512 ) ")
print(angle_rates)



## multiply 2 vectors to get a matrix of size [40, 512]
angle_rads = pos * angle_rates


print("angle_rads = pos * angle_rates")
print(angle_rads)

print("i.shape ", i.shape)
print("pos.shape ", pos.shape)
print("batch.shape ", batch.shape)
print("angle_rates.shape ", angle_rates.shape)
print("angle_rads.shape ", angle_rads.shape)


angle_rads[:, 0::2]    = np.sin(angle_rads[:, 0::2])       ## even index
angle_rads[:, 1::2]    = np.cos(angle_rads[:, 1::2])       ## odd index

print("sin and cos to angle_rads ")
print(angle_rads)

plt.pcolormesh(angle_rads, cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()

angle_rads = angle_rads[np.newaxis, ...]   ## (1, 40, 512)   for broadcasting?
print("angle_rads[np.newaxis, ...]  ")
print("angle_rads.shape ", angle_rads.shape)

'''

    
    i.shape  (1, 512)
    pos.shape  (40, 1)
    batch.shape  (64, 40, 512)
    angle_rates.shape  (1, 512)
    angle_rads.shape  (40, 512)
    sin and cos to angle_rads
    
    angle_rads[np.newaxis, ...]
    angle_rads.shape  (1, 40, 512)


'''


print("****************************")

#########################################################

print("<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")


