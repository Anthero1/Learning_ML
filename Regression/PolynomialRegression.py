import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.figure import Figure
import numpy as np
import random
import math


#These parameters describe how the target dataset will be generated
#thetas are the coefficients, in ascending order (e.g. intercept, x-coefficient, x^2-coefficient, etc.)
#xrange and leftXbound describe which portion of the curve you want your data to fit around
#noise is the variation (absolute value, not percentage) in the generated dataset from the curve described by the thetas
noise=0.2
xrange = 4
leftXBound = -2
targetThetas = [200,0,1,0,-0.5]


#change these values to tune your learning
#cycles is the number of times regression is done
#degree is which degree you want the output of the learn to be
#a is the learning rate. If the program fails to produce any curve, a is too large.
#if the programs doesn't match the target curve well enough, either increase a or increase the number of cycles
cycles=5000
degree = 5
a = 0.002



#calculates the y-coordinate of a point on the curve, based on the x-coordinate of said point
def poly(thetas, x):
    ans=0
    for i in range(len(thetas)):
        ans+=thetas[i]*(x**i)
    return ans


#Calculates the cost of the current estimate through Sum((PredY-ActualY)^2)/(2*n)
def cost(xs, ys, thetas):
    ans = 0
    denominator=2*len(xs)
    for i in range(len(xs)):
        try:
            ans+=(((poly(thetas, xs[i])-ys[i]))**2)/denominator
        except:
            return -1
    return ans


#calculates the partial derivatives of the cost function, with respect to each coefficient of the curve
def derivatives(thetas,xs,ys):
    derivs = []
    for i in range(len(thetas)):
        temp = 0
        for j in range(len(xs)):
            temp += (poly(thetas, xs[j])-ys[j])*(xs[j]**(i))/len(xs)
        derivs.append(temp)
    return derivs


#adjusts the thetas prediction, based on the partial derivatives of the cost function
def descent(thetas,ders,a):
    for i in range(len(thetas)):
        thetas[i]=thetas[i]-(a)*ders[i]
    return thetas


#main regression function
def regression(cycles, a, xs, ys, degree):

    #initializes random starting variables
    thetas = [random.randint(0,10)]
    for i in range(degree):
        thetas.append(random.randint(0,10))

    #initializes history tracking
    history=[]
    for i in range(cycles):
        history.append([[],1])

    #runs the regression function a certain number of times (decided by the "cycles" variable)
    for i in range(cycles):

        #Calculates and stores updated predictions and the cost of those predictions
        temp = descent(thetas,derivatives(thetas,xs,ys),a)
        for j in range(degree+1):
            history[i][0].append(temp[j])
        history[i][1] = cost(xs, ys, history[i][0])

        #Live output, uncomment to see how the cost changes over time. Good if you need to tune the learning rate.
        if i%100 == 0:
            print(history[i][1])

        #updates the predicted values
        thetas = history[i][0]

        #increases or decreases the learning rate based on whether the newest prediction was better or worse than the previous
        if i > 0 and history[i][1] > history[i-1][1]:
            a=a*0.99
        else:
            a=a*1.001

    return history


#generates data to regress over using the parameters chosen by the user
xs=[]
ys=[]
for i in range(100):
    i=i*(xrange/100)+leftXBound
    q=poly(targetThetas, i)
    xs.append(i)
    ys.append(random.uniform(q-noise,q+noise))
    i=(i-leftXBound)/(xrange/100)


#calls the main regression function
history = regression(cycles,a,xs,ys,degree)
print(history[cycles-1][0])


#creates the line of best fit points, used for plotting
lineOfBestFitxs = []
lineOfBestFitys = []
for i in range(30):
    lineOfBestFitxs.append(min(xs)+(max(xs)-min(xs))*(i+1)/30)
    lineOfBestFitys.append(poly(history[cycles-1][0], lineOfBestFitxs[i]))


plt.figure(figsize =(16, 6))

#creates the regression history plot
plt.subplot(131)
for j in range(degree+1):
    storage = []
    for i in range(cycles):
        storage.append(history[i][0][j])
    label="Theta"+str(j)+": "+str(round(history[cycles-1][0][j],1))
    plt.plot(storage, label=label)
plt.legend()
plt.title('Coefficient Regression History')

#creates the results plot
plt.subplot(132)
plt.plot(xs, ys, 'bo')
plt.plot(lineOfBestFitxs,lineOfBestFitys, color="red")
plt.title('Regression Result')

#creates the cost function history plot
plt.subplot(133)
temp = []
for i in range(len(history)):
    temp.append(history[i][1])
plt.semilogy(temp)
#plt.plot(temp)
#plt.axis([0,cycles,0,history[100][1]])
plt.title('Cost Function History')

plt.show()