import math

def add(x,y,f):
    return f(x)+f(y)

print("\nadd(25,9,math.sqrt):\n",add(25,9,math.sqrt))