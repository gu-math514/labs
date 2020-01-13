# -*- coding: utf-8 -*-

# lab2
 
#Exercise 1
# write a function that takes as input mean and standard deviation and returns 2 functions
# 1. a function that given an integer returns samples from a normal distribution
# 2. a function that given x returns the normal pdf value at x
# Note: this design ensures that both functions are using the same mean/std
# call your function and plot the histogram of a sample and overlay it with the normal density curve
# you might want to use the numpy function linspace
 

#Exercise 2
# write a joint density function (function of x,label) for a mixture of two normal distributions 
# Use the joint density to compute a marginal density for x
# Plot the marginal and the class conditional density curves p(x|Ci)
    


#Exercise 3
# write a function to simulate to results of tossing a coin n times with probability p of heads
# run 100 experiments to toss a fair coin 15 times and store the mean number of heads for each experiment
# plot a histogram of the means and overlay with an approximating normal density of the sampling distribution
# the sample distribution has std=sigma/sqrt(n), where sigma is the std of the trial and n is the number of trials

 
#Exercise 4
# write a closure that accepts, x,a,b and returns 2 functions, a function f that returns (x+a)*(x*x-b*b) and function g that is its derivative,
# use scipy.optimize brentq root finder to find smallest root of given function with a=1., b=1.1.
# plot the function f and derivative g funct
# Based on the plot you might randomly try a window of -2 and 0 and see what happens.

# One approach to find a reasonable window is to find the roots of the derivative, 
# as those are inflection points, to comple up with reasonable windows to find the negative roots.
# Use root finder to find roots of the derivative function and use that to find a suitable window to find the 2 negative roots of f
 




# Exercise 5
# for pairs of x,y points compute the signed perpindicular distance to a discriminant line with parameters b,w
# the equation of the discriminant is b+x*w[0]+y*w[1] = 0
# perpindicular distance is the shortest distance between a point and line
# use  a simple example to show distance is correct and show distance is signed
# show that changing the sign is working by testing a point above and below the x-axis with a horizontal line

 
