# I'm using vpython for python 3, which works a little differently from python 2
from vpython import *
import matplotlib.pyplot as plt
import math
import numpy

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
maxVStep = 10**6/100

# a clock that ticks upward, keeping track of real time
realTime = 0
# how many seconds the simulation runs for
endTime = 20

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
    # rate(.5/dt)

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
        # to account for curvature and "now + dt," which overaccounts for curvature
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

# Kepler's 2nd law states that the area swept out over a set period of time is the same no matter what stage of orbit the body is in
# you recall that I used a variable time-step, which means that I have some fun sub-triangle excitement ahead of me

# the lighter mass was the 2nd one in the list
movingMass = masses[1]

# transforms a vpython vector to an array that can be used with numpy
def vecToArr(vec):
    return [vec.x, vec.y, vec.z]

# a function that gives the area of a triangle sector shape defined by a center
# at origin, a point at histObj1, and a point at histObj2. Each history object
# has a value of the derivative (symmetric velocity) and the time at which the mass
# was at that position. If one of those times is less than maxT, then this function
# calculates the partial area of the triangle shape
def partialArea(histObj1, histObj2, origin, maxT):
    (p1, p2) = (histObj1['pos'], histObj2['pos'])

    # determine if we should calculatee a partial area or not
    partialArea = False
    # helpful info if we do need to calculate a partial area
    partialTriangleP2 = None
    partialTravel = None
    # if our second point is too far in the future, calculate a partial area
    if histObj2['t'] > maxT:
        partialArea = True
        # partialTravel gives a sense of how far from histObj1 maxT is (from 0 to 1)
        partialTravel = (maxT - histObj1['t']) / (histObj2['t'] - histObj1['t'])
        # linearly interpolates a point between p1 and p2 based on the partialTravel
        partialTriangleP2 = p1 * (1 - partialTravel) + partialTravel * p2

    # calculate triangle area using the cross product. Recall that the cross product gives
    # the area of the parallelogram defined by v1 and v2, thus we divide by two to get the triangle
    triangleArea = 0
    if partialArea:
        triangleArea = mag(cross(p1, partialTriangleP2)) / 2
    else:
        triangleArea = mag(cross(p1, p2)) / 2

    # now compute the sector-ish area on the outside of the triangle (to see how much it makes a difference)
    # to do this, we need to transform our triangle shape to useful coordinates.
    # we'll use the vector from p1 to p2 for one basis vector and go from there
    basis1 = (p2 - p1) / mag(p2 - p1)
    radius = .5*(p1+p2) - origin
    basis2 = cross(basis1, radius)
    basis2 /= mag(basis2)
    basis3 = cross(basis1, basis2)

    # now to transform to standard basis, we apply the inverse of the linear tranformation given by the basis vectors above
    transformation = numpy.column_stack([vecToArr(basis1), vecToArr(basis2), vecToArr(basis3)])
    inverseTransformation = numpy.linalg.inv(transformation)
    def transform(vec):
        return [x[0] for x in numpy.dot(inverseTransformation, numpy.column_stack([vecToArr(vec)]))]
    normDeriv1 = transform(histObj1['sym_der'])
    normDeriv2 = transform(histObj2['sym_der'])
    normOffset = transform(p2 - p1)

    # now we have a transformed basis, where p2 - p1 points in the positive x direction and the velocity
    # vectors point in either the positive or negative z direction. Now we can drop the y component, effectively
    # projecting the velocity vectors onto the plane formed by the origin and the 2 position pooints

    # we model this as a quadratic function. Normally I would do a sector area, but that involves calculating the
    # radius of a circle, which diverges as the vectors approach pointing the same direction. This (I think) would
    # mean that this would be an unstable calculation.

    aveDerivX = .5*(abs(normDeriv1[0]) + abs(normDeriv2[0]))
    aveDerivY = .5*(abs(normDeriv1[2]) + abs(normDeriv2[2]))

    # our quadratic function is given by y = (aveDerivY / (aveDerivX * |normOffset|)
    # integrating from 0 to b, we arrive at (1/2)(dery/derx)b^2 - (1/3)(dery/(derx*|normOffset|))*b^3
    # we can approximate b as |normOffset| * partialTravel, if we need to break up our triangle
    b = math.sqrt(normOffset[0]**2 + normOffset[1]**2)
    normOffset = b
    if partialArea:
        b *= partialTravel
    quadraticArea = .5*(aveDerivY / aveDerivX)*b**2 - (1.0/3.0)*(aveDerivY /  (aveDerivX * normOffset))*b**3

    return triangleArea + quadraticArea


# generates helper data for the partialArea function
for i in range(1, len(movingMass.history) - 1):
    [prevObj, obj, nextObj] = [movingMass.history[j] for j in [i-1,i,i+1]]
    # calculate the symmetric derivative (the average of the left and right derivatives)
    obj['sym_der'] = .5*(obj['pos'] - prevObj['pos']) / (obj['t'] - prevObj['t']) + .5*(nextObj['pos'] - obj['pos']) / (nextObj['t'] - obj['t'])

timeStep = baseDt
areas = []
# test a bunch of initial positions, sweeping out the area travelled in timeStep
for i in range(1, len(movingMass.history) - 1, 10):
    # this is the starting point of the swept area
    obj = movingMass.history[i]
    startTime = obj['t']
    endTime = startTime + timeStep
    # keeps track of the next index past nextObj
    nextObjIndex = i + 1

    nextObj = movingMass.history[nextObjIndex]
    nextObjIndex += 1
    area = 0
    # first condition so that we don't run out of elements in our list
    while nextObjIndex < len(movingMass.history) and obj['t'] < endTime:
        area += partialArea(obj, nextObj, masses[0].history[nextObjIndex-1]['pos'], endTime)
        obj = nextObj
        nextObj = movingMass.history[nextObjIndex]
        nextObjIndex += 1

    # only use completed area segments
    if(nextObjIndex < len(movingMass.history)):
        print(i,len(movingMass.history),area)
        areas.append(area)

plt.plot(areas)
plt.show()
