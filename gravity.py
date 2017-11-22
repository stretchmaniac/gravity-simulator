# I'm using vpython for python 3, which works a little differently from python 2
from vpython import *
import matplotlib.pyplot as plt
import math
from functools import reduce
import numpy as np
from numpy.linalg import eig, inv

# initialize some test masses, roughly in proportion to the sun/earth system
m1 = sphere(pos=vector(0,0,1), radius=10**8/1000, color=color.blue, mass=2*10**30)
# set an initial velocity
m1.vel = vector(0, 0, 0)

# and a second sphere
m2 = sphere(pos=vector(2*10**8/50,0,-2), radius = 10**8/1000, color=color.red, mass=6*10**24)
m2.vel = vector(0,-.5*10**7, -.5*10**6)

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
endTime = 5

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
    #plt.show()

# Kepler's 3rd law states that the square of the period is proportional to the cube of the radius of orbit
h = masses[1].history
startPoint = h[0]
# find the point closest to the initial position by comparing the cumulative "best" distance to each point in succession
endIndex = reduce(lambda x,y: x if mag(h[x]['pos']-startPoint['pos']) < mag(h[y]['pos']-startPoint['pos']) else y, range(1, len(masses[1].history)))
endPoint = h[endIndex]

# do some interpolation
pts = [h[endIndex + 1], h[endIndex - 1]]
bestPt = None
bestT = 0
for p in pts:
    # this would really be better in binary search form (since this is a really smooth and predictable function),
    # but since efficiency isn't really important here I don't care that much
    precision = 1000
    # between 0 and 1, in increments of 1/1000
    for t in [x/precision for x in range(precision)]:
        # interpolate between the point and its neighbor
        newPt = endPoint['pos'] + t*(p['pos'] - endPoint['pos'])
        # if this interpolated point is closer than any point previously, then use that point
        if bestPt == None or mag(newPt - startPoint['pos']) < mag(bestPt - startPoint['pos']):
            bestPt = newPt
            bestT = endPoint['t'] + t*(p['t'] - endPoint['t'])

period = bestT - startPoint['t']
# quantify how close we were to the initial starting point, with the second value for scale
print('period dist error: ',mag(bestPt - startPoint['pos']), mag(bestPt))

# now we get to calculate the eccentricity of our ellipse...
# I'm going to copy some code from kepler's first law, if you don't mind

# found from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
# I currently do not know enough about eigenvalues to explain this -- ask me in a few months
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

# from the same source as fitEllipse
def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

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
# b2 and b3 lie in the plane, while b1 is normal
b2 = vector(c, -b, 0)
b2 /= mag(b2)
b3 = cross(b1, b2)

# turns a vpython vector into an array for use with numpy
def toArr(vec):
    return [vec.x, vec.y, vec.z]

# the transformation from the standard basis vectors to the orthonormal basis we
# just computed is the new basis vectors as columns of a square matrix
invTransformation = np.column_stack([toArr(b1), toArr(b2), toArr(b3)])
# the inverse transformation is the inverse of the matrix representing the transformation
transformation = inv(invTransformation)

# transform all of our points by multiplying transformation by the points as column vectors
transformed = [np.transpose(np.dot(transformation, np.array([[p.x],[p.y],[p.z]])))[0] for p in points]

# now get the y and z components of the transformed points (recall that b2 and b3 lie in the plane and b1 is normal),
# and feed them into the fit ellipse method
xyzs = np.column_stack(transformed)
args = fitEllipse(xyzs[1], xyzs[2])

axes = ellipse_axis_length(args)

print('period: ', period)
print('axes: ', axes)
# use the semi-major axis
meanAxis = max(*axes)
print(period**2/(meanAxis**3))
print(4*math.pi**2/(G*masses[0].mass))

# ouput:
'''
simulated time will be 5 seconds.
period dist error:  8.429010193966816 3989799.9668614417
period:  3.1388830257312073
axes:  [ 3217397.8952403   3120772.54456622]
2.95825728625e-19
2.9582529126139497e-19
'''
