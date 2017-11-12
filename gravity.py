# I'm using vpython for python 3, which works a little differently from python 2
from vpython import *
import matplotlib.pyplot as plt
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
m2 = sphere(pos=vector(10**8/50,0,-2), radius = 10**8/1000, color=color.red, mass=6*10**24)
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
maxVStep = 10**6/10000

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
    rate(.5/dt)

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

focus1 = masses[0].pos
focus2 = focus1
points = [x['pos'] for x in masses[1].history][0:-1:1000]

reasonableDist = mag(masses[0].pos - masses[1].pos)
dx = reasonableDist / 1000000
dy = dx
dz = dx

prevVars = [ellipseVariance(points, focus1, focus2) + 100 for i in range(10)]
gradMultiplier = 1
for i in range(5000):
    print('minimizing '+str(i)+' out of 5000')
    var = ellipseVariance(points, focus1, focus2)
    dxVar = (ellipseVariance(points, focus1, focus2 + vector(dx, 0, 0)) - var) / dx
    dyVar = (ellipseVariance(points, focus1, focus2 + vector(0, dy, 0)) - var) / dy
    dzVar = (ellipseVariance(points, focus1, focus2 + vector(0, 0, dz)) - var) / dz
    # follow the gradient
    gradVec = -vector(dxVar, dyVar, dzVar)
    if mag(gradVec) > reasonableDist / 1000:
        gradVec *= reasonableDist / (1000 * mag(gradVec))

    prevVarAve = 0
    for j in prevVars:
        prevVarAve += j
    prevVarAve /= len(prevVars)

    if var > prevVarAve:
        gradMultiplier *= .75

    if i > 500 and i % 100 == 0:
        gradMultiplier *= .85

    focus2 += gradMultiplier*gradVec

    prevVar = var
    print(var, focus2)

variance = ellipseVariance(points, focus1, focus2)
print('total variance:', var)
print('variance per point: ', var / len(masses[0].history))
print('focus position 1: ', focus1)
print('focus position 2: ', focus2)
print('number of points: ', len(points))

# output:
'''
total variance: 14367798.033235407
variance per point:  15.064185186014484
focus position 1:  <15.228077, -292.574737, -13.628760>
focus position 2:  <-4268656.238315, -54223.364443, -956.278581>
number of points:  954
'''
