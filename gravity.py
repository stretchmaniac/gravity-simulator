# I'm using vpython for python 3, which works a little differently from python 2
from vpython import *
import matplotlib.pyplot as plt
import math

# initialize some test masses, roughly in proportion to the sun/earth system
m1 = sphere(pos=vector(0,0,0), radius=10**8/1000, color=color.blue, mass=2*10**30)
# set an initial velocity
m1.vel = vector(0, 0, 0)

m2 = sphere(pos=vector(10**8/75,0,0), radius = 10**8/1000, color=color.red, mass=6*10**24)
m2.vel = vector(0,-10**7, -10**6)

m3 = sphere(pos = vector(-10**8/75, 0,0), radius=10**8/1000, color=color.yellow, mass=6*10**24)
m3.vel = vector(5*10**6,10**7, -2*10**6)

m4 = sphere(pos = vector(-10**8/50, 0,0), radius=10**8/1000, color=color.green, mass=6*10**24)
m4.vel = vector(-0,1*10**7, -2*10**5)

# for fun, we extend this to an arbitrary number of masses
masses = [m1, m2, m3, m4]
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
baseDt = .01
dt = baseDt

# caps the maximum change in position by changing dt accordingly (see below)
maxPStep = 10**6/100
# caps the maximum change in velocity also
maxVStep = 10**6/100

# how many seconds the simulation runs for
realTime = 0
endTime = 10

# the gravitational constant, G
G = 6.67259*10**(-11)

print('simulated time will be '+str(endTime)+' seconds.')

# this plots the xy position of the masses
# note that previous versions of vpython, this was "gdisplay"
trajplot = graph(x=0, y=0, width=800, height=400, title='y vs x',
    xtitle='x position', ytitle='y position', xmin=-5.5, xmax=5, ymin=0, ymax=25, foreground=color.black, background=color.white)

def getAcceleration(m1, pos):
    fNet = vector(0,0,0)
    for m2 in masses:
        if m1 == m2:
            continue
        fMagnitude = G*m1.mass*m2.mass/dot(m1.pos-m2.pos, m1.pos-m2.pos)
        # the norm returns a unit function, in this case r_hat
        fNet += fMagnitude * norm(m2.pos-m1.pos)
    return fNet / m1.mass

# this was changed from a for loop to a while loop to keep track of time instead of steps
while realTime < endTime:
    # we're hoping for this to run at real time.
    # Unfortunately this is a bit optimistic. We'll settle for a tenth time.
    # Thus this frame should take 10*dt seconds, so the rate is .1/dt
    rate(.5/dt)

    # sum up the kinetic and potential energy as we do other things
    PE = 0

    for m in masses:
        # initializing a net force vector with magnitude zero
        m.fNet = vector(0,0,0)

    # iterates through all possible pairs of points once
    for i in range(len(masses)):
        for j in range(i+1, len(masses)):
            # shorthand for two separate assignments
            (m1, m2) = (masses[i], masses[j])
            # Newton's law of gravity equates force to G m1 m2 / (r.r) r_hat
            # we will first compute G m1 m2 / (r.r)
            fMagnitude = G*m1.mass*m2.mass/dot(m1.pos-m2.pos, m1.pos-m2.pos)
            # the norm returns a unit function, in this case r_hat
            m1.fNet += fMagnitude * norm(m2.pos-m1.pos)
            m2.fNet += fMagnitude * norm(m1.pos-m2.pos)

            # the potential energy between two objects is given by -G m1 m2 / r
            PE -= G*m1.mass*m2.mass/mag(m1.pos - m2.pos)

    # update the potential energy history
    potentialEnergyHistory.append(PE)

    # we see a breakdown of precision when v*dt or a*dt is sufficiently large. Thus we want to cap
    # dt so that v*dt < maxVStep and a*dt < maxAStep for all v and a
    maxVel = 0
    maxAcc = 0
    for m in masses:
        # calculate a as F_net / mass
        m.acc = m.fNet / m.mass
        # update velocity
        m.vel += m.acc * dt
        # update position
        m.pos += m.vel * dt

        # add the position to the trail
        m.trail.append(pos=m.pos)

        # update our mass histories
        m.history.append({
            't':realTime,
            'pos':m.pos,
            'vel':m.vel,
            'acc':m.acc
        })

        # update the maximum velocity and acceleration if needed
        if mag(m.vel) > maxVel:
            maxVel = mag(m.vel)
        if mag(m.acc) > maxAcc:
            maxAcc = mag(m.acc)

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
