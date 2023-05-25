import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.figure import Figure
import numpy as np
import random
import math


#Calculates the cost through Sum((PredY-ActualY)^2)/(2*n)
def cost(xs, ys, m, b):
    ans = 0
    denominator=2*len(xs)
    for i in range(len(xs)):
        try:
            ans+=(((m*xs[i]+b-ys[i]))**2)/denominator
        except:
            return -1
    return ans

#calculates the partial derivatives of the cost function with respect to the slope (derM) and the y-intercept (derB)
def derivatives(m,b,xs,ys):
    derM = 0
    derB = 0
    for i in range(len(xs)):
        derM += (m*xs[i]+b-ys[i])*xs[i]
        derB += (m*xs[i]+b-ys[i])
    derM = derM/len(xs)
    derB = derB/len(xs)
    return [derM,derB]

#does gradiant descent by adjusting the m/b values according to the results of the derivatives() function
def descent(m,b,ders,a):
    newM = m-a*ders[0]/10
    newB= b-a*ders[1]
    return newM,newB


#main regression function
def regression(cycles, a, xs, ys):

    #initializes random starting variables
    m = random.randint(-10,10)
    b = random.randint(-300,500)

    #initializes history tracking
    history=[[m],[b], [cost(xs, ys, m, b)]]

    #runs the regression function a certain number of times
    for i in range(cycles):

        #Calculates and stores updated values for m/b
        temp1, temp2 =descent(m,b,derivatives(m,b,xs,ys),a)

        #stores the new m/b values and the new cost
        history[0].append(temp1)
        history[1].append(temp2)
        history[2].append(cost(xs,ys, temp1, temp2))

        # if(i%10)==0:
        #     print(i,"-",temp1,"-",temp2)

        #updates the m/b values
        m=temp1
        b=temp2

    return history


#generates semi-random data to regress over, but makes sure the data is corellated
xs=[]
ys=[]
for i in range(130):
    xs.append(random.uniform(i-20,i+20))
    ys.append(random.uniform(120-i,80-i))


#number of regression cycles
cycles=30000

#calls the main regression function
history = regression(cycles,0.00034,xs,ys)

#creates the line of best fit points, for plotting
m=history[0][cycles]
b=history[1][cycles]
lineOfBestFitxs=[min(xs),max(xs)]
lineOfBestFitys=[m*min(xs)+b,m*max(xs)+b]

plt.figure(figsize =(18, 4))

#creates the regression history plot
plt.subplot(131)
plt.plot(history[0], '--', label='Slope History', color='orange')
plt.plot(history[1], '--', label='Y-intercept History', color='blue')
plt.legend()
plt.title('Regression History')

#creates the results plot
plt.subplot(132)
plt.plot(lineOfBestFitxs,lineOfBestFitys)
plt.plot(xs, ys, 'bo')
plt.title('Regression Result')

#creates the cost function history plot
plt.subplot(133)
plt.plot(history[2])
plt.axis([0,cycles,0,1000])
plt.title('Cost Function History')

plt.show()