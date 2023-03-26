import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

def cost(xs, ys, m, b):
    ans = 0
    for i in range(xs):
        ans+=(m*xs[i]+b-ys[i])**2
    return ans/(2*len(xs))

def derivatives(m,b,xs,ys):
    derM = 0
    derB = 0
    for i in range(len(xs)):
        derM += (m*xs[i]+b-ys[i])*xs[i]
        derB += (m*xs[i]+b-ys[i])
    derM = derM/len(xs)
    derB = derB/len(xs)
    return [derM,derB]

def descent(m,b,ders,a):
    newM = m-a*ders[0]
    newB= b-a*ders[1]
    return newM,newB

def regression(cycles, a, xs, ys):
    m = random.randint(-5,5)
    b = random.randint(-30,30)
    for i in range(cycles):
        temp1, temp2 =descent(m,b,derivatives(m,b,xs,ys),a)
        if i%10==0:
            print(i,"-","Old m/b:",m," ",b,"    new m/b:",temp1,"   ",temp2)
        m=temp1
        b=temp2
    return m,b

xs = [1,2,3,4,5,6,7,8,9,10]
ys = [11,8,8,7,4,7,4,3,3,1]

m,b = regression(1000,0.5,xs,ys)

linexs=[min(xs),max(xs)]
lineys=[m*min(xs)+b,m*max(xs)+b]
plt.plot(xs, ys, 'bo')
plt.plot(linexs,lineys)

plt.show()