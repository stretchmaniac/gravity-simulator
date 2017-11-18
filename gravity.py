# I'm using vpython for python 3, which works a little differently from python 2
from vpython import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv,eig
import math
from functools import reduce

# Kepler's first law states that the planets follow an elliptical path around the sun, where the
# the sun is one focus of the ellipse.
# We model this as a massive object with zero initial velocity and a much less massive object orbiting it

# initialize some test masses, roughly in proportion to the sun/earth system
m1 = sphere(pos=vector(0,0,1), radius=10**8/1000, color=color.blue, mass=2*10**30)
# set an initial velocity
m1.vel = vector(0, 0, 0)

# and a second sphere
m2 = sphere(pos=vector(10**8/50,0,-2*10**5), radius = 10**8/1000, color=color.red, mass=6*10**24)
m2.vel = vector(0,-1.0*10**7, -.5*10**6)

# for fun, we extend this to an arbitrary number of masses
masses = [m1, m2]
for m in masses:
    # the trial is the line that marks the mass's path
    # to make it run faster, we restrict the trail length to a reasonable length
    m.trail = curve(color=m.color, retain=5000)
    # keeps a log of the mass's position over time in a dictionary with keys time 't',
    # position 'pos', velocity 'vel', and acceleration 'acc'
    m.history = []

# Aligns with elements in history for each mass. This tracks the total potential energy of the system
potentialEnergyHistory = []

# delta time (seconds)
# note that this is not necessarily constant
baseDt = .1
# since dt changes, we want a baseline (baseDt)
dt = baseDt

# caps the maximum change in position by changing dt accordingly (see below)
maxPStep = 10**6/10
# caps the maximum change in velocity also
# fyi, this is very high precision. This yeilds a total change in energy from beginning to end
# of about .00000013%
maxVStep = 10**6/1000

# a clock that ticks upward, keeping track of real time
realTime = 0
# how many seconds the simulation runs for
endTime = 10

# the gravitational constant, G
G = 6.67259*10**(-11)

print('simulated time will be '+str(endTime)+' seconds.')

# calculates the acceleration due to the gravitational field of the other masses
def numericAcceleration(m):
    # sum of forces due to each mass
    fNet = vector(0,0,0)
    for other in masses:
        # we don't want to compute the gravitational field due to itself!
        if m != other:
            # F = g m1 m2 / r**2 r_hat
            # note that this uses the property bestApproxR, which is explained farther down
            fNet += G * m.mass * other.mass / mag(m.bestApproxR - other.bestApproxR)**2 * norm(other.bestApproxR - m.bestApproxR)
    # a = f_net / m
    return fNet / m.mass

# this was changed from a for loop to a while loop to keep track of time instead of steps
while realTime < endTime:
    # we're hoping for this to run at real time.
    # Unfortunately this is a bit optimistic. We'll settle for a tenth time.
    # Thus this frame should take 10*dt seconds, so the rate is .1/dt
    #rate(.5/dt)

    # we will use an iterative approach to better approximate the change in velocity and position
    for m in masses:
        # this is our best guess for the position of m dt seconds from now. This will change as we
        # get a better idea of what's going on
        # we're doing this in a separate loop so that the numericAcceleration function will work in the next loop
        m.bestApproxR = m.pos

    for m in masses:
        # calculate the change in velocity as we would normally. This is the first order approximation
        tempA = numericAcceleration(m)
        # dv / dt = a, so an approximation of dv is a * dt
        tempV = m.vel + tempA*dt

        # this is our best guess for the acceleration of m dt seconds from now
        m.bestApproxA = tempA
        # this is our best guess for the velocity of m dt seconds from now
        m.bestApproxV = tempV
        # our best guess for the position of m follows from the first order approximation:
        # dr / dt = v ==> dr = v * dt
        m.bestApproxR = m.pos + tempV*dt
        # the acceleration at this instant (before dt)
        m.acc = tempA

    for iterations in range(10):
        # use v(t + dt) ~ v(t) + dt (a(t) + a(t + dt)) / 2
        #     r(t + dt) ~ r(t) + dt (v(t) + v(t + dt)) / 2
        #     a(r + dt) ~ numericAcceleration(r(t + dt))
        # this is a cyclical set of relations that uses the average derivative at "now," which fails
        # to account for curvature and at "now + dt," which overaccounts for curvature
        for m in masses:
            # calculate the acceleration dt seconds from now using the best guess (future) positions of all the masses
            m.bestApproxA = numericAcceleration(m)
            # calculate the best guess future velocity and position according to the above formula
            m.bestApproxV = m.vel + dt * (m.acc + m.bestApproxA) / 2
            # we use a temporary variable as to not change m.bestApproxR for the rest of the masses in this iteration
            m.bestApproxRTemp = m.pos + dt * (m.vel + m.bestApproxV) / 2
        for m in masses:
            # update the best guess future position from the temporary variable
            m.bestApproxR = m.bestApproxRTemp

    # we see a breakdown of precision when v*dt or a*dt is sufficiently large. Thus we want to cap
    # dt so that v*dt < maxVStep and a*dt < maxAStep for all v and a
    maxVel = 0
    maxAcc = 0
    for m in masses:
        # make the position and velocity our best guess
        m.pos = m.bestApproxR
        m.vel = m.bestApproxV

        # add the position to the trail
        m.trail.append(pos=m.pos)

        # update our mass histories with some metadata
        m.history.append({
            't':realTime,
            'pos':vector(m.pos.x, m.pos.y, m.pos.z),
            'vel':vector(m.vel.x, m.vel.y, m.vel.z),
            'acc':vector(m.acc.x, m.acc.y, m.acc.z)
        })

        # update the maximum velocity and acceleration if needed
        if mag(m.vel) > maxVel:
            maxVel = mag(m.vel)
        if mag(m.acc) > maxAcc:
            maxAcc = mag(m.acc)

    # for analytical purposes, I want to keep track of the potential energy of the system.
    # this is the sum of the potential energy of each combination of masses
    PE = 0
    for i in range(len(masses)):
        # don't repeat mass combinations
        for j in range(i+1, len(masses)):
            # short hand for two assignments
            (m1, m2) = (masses[i], masses[j])
            # PE = - G m1 m2 / r
            PE += -G * m1.mass * m2.mass / mag(m1.pos - m2.pos)
    # add to the potential energy log. This increments along with each mass's history property
    potentialEnergyHistory.append(PE)

    # reset dt to the baseline
    dt = baseDt
    # adjust dt as the velocity and acceleration of the masses increase, if needed
    if maxVel*dt > maxPStep:
        # now maxVel*dt == maxPStep
        dt = maxPStep / maxVel
    if maxAcc*dt > maxVStep:
        # now maxAcc*dt == maxVStep
        dt = maxVStep / maxAcc

    # update the real time by dt
    realTime += dt

# now make a pretty graph showing potential and kinetic energy as a function of time for the system
if len(masses) > 0:
    # all the masses have the same time information
    ts = [x['t'] for x in masses[0].history]
    # kinetic energy is 1/2 mv^2, summed over all the masses
    KEs = []
    # for each step...
    for i in range(len(masses[0].history)):
        KE = 0
        # sum up the kinetic energies of each mass
        for m in masses:
            velVec = m.history[i]['vel']
            # a.a == |a|^2
            KE += .5*m.mass*dot(velVec, velVec)
        # ... and add it to the list
        KEs.append(KE)
    PEs = potentialEnergyHistory

    # plot total energy as well
    totals = [KEs[i] + PEs[i] for i in range(len(masses[0].history))]

    plt.plot(ts, KEs)
    plt.plot(ts, PEs)
    plt.plot(ts, totals)
    plt.show()

# we now need to check how elliptical the path was
# we know that the sun is one focus (let's call it a point a). We seek another point b such that
# |p - a| + |p - b| has the least varience for all points p in our data set
# we will use a gradient descent style algorithm for simplicity

def ellipseVariance(pts, focus1, focus2):
    dists = [mag(x - focus1) + mag(x - focus2) for x in pts]
    mean = 0
    for d in dists:
        mean += d
    mean /= len(dists)

    var = 0
    for d in dists:
        var += (d - mean)**2
    return var

# found from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

focus1 = masses[0].pos
focus2 = focus1
points = [x['pos'] for x in masses[1].history]

# transform to xy basis. This means finding the best plane approximation
# of the points and projecting the points onto the plane
# We use a matrix formulation for a linear regression. Please ask me if you want a derivation!
X = np.array([[1, p.x, p.y] for p in points])
Z = np.array([[p.z] for p in points])

coeffs = np.dot(np.dot(inv(np.dot(np.transpose(X),X)), np.transpose(X)), Z)
(a,b,c) = (coeffs[0][0], coeffs[1][0], coeffs[2][0])

# now we need an orthonormal basis for the plane. We have that our plane
# a + bx + cy = z has a normal vector (b,c,-1). We see that (c, -b, 0) is perpendicular to this
# and we can use cross product to find the last vector
b1 = vector(b, c, -1)
b1 /= mag(b1)
# b2 and b3 lie in the plane
b2 = vector(c, -b, 0)
b2 /= mag(b2)
b3 = cross(b1, b2)

def toArr(vec):
    return [vec.x, vec.y, vec.z]

invTransformation = np.column_stack([toArr(b1), toArr(b2), toArr(b3)])
transformation = inv(invTransformation)

# transform all of our points
transformed = [np.transpose(np.dot(transformation, np.array([[p.x],[p.y],[p.z]])))[0] for p in points]

xyzs = np.column_stack(transformed)
args = fitEllipse(xyzs[1], xyzs[2])
rawCenter = ellipse_center(args)

# transform back to normal coordinates
v = np.array([[0], [rawCenter[0]], [rawCenter[1]]])
ellipseCenter = [x[0] for x in np.dot(invTransformation, v)]
ellipseFocusVec = vector(focus1.x + 2*(ellipseCenter[0]-focus1.x), focus1.y + 2*(ellipseCenter[1]-focus1.y), focus1.z + 2*(ellipseCenter[2] - focus1.z))
print('focus: ',ellipseFocusVec)
var = ellipseVariance(points, focus1, ellipseFocusVec)
print('total variance: ', var)
print('variance per point: ', str(var/len(points)))
