square = lambda x:x**2
print(square(2))
sum = lambda  x, y : x + y
print(sum(2, 3))  # 5


l = [1,2,3,5,-9,0,45,-99]
print(list(filter(lambda x:x < 0,l)))

from functools import reduce
l = [1,2,3,5,-9,0,45,-99]
print(reduce(lambda x,y:x+y,l))

for k in range(1,10):
    print(k)
