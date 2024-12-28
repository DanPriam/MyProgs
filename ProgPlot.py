# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 22:26:15 2022

@author: Даниил
"""
from numpy import zeros, linspace, sin, pi, complex64, real
from matplotlib.pyplot import style, figure, axes
from matplotlib import pyplot as plt
from celluloid import Camera
from  numpy import savez, load

results = load('Results_M200_N200_St200.npz')
x = results['x']
Steps = results['Steps']
Q = results['Q']
UU = results['UU']
q = sin(3*pi*x)

fig1 = plt.figure()
camera = Camera(fig1)
for i in range( Steps):
    plt.plot(x,Q[i],color='r')
    plt.plot(x,q, linestyle = '--',color='g')
    camera.snap()
animation = camera.animate()
#ax2 = axes(xlim=(0,1), ylim=(-1.2,1.2))
#ax2.plot(x,Q[Steps-1],color='r')
#ax2.plot(x,q, linestyle = '--',color='g')
#ax2.grid(True)
animation.save('Video_M200_N200_St200.gif')
#style.use('white_background')
fig = figure()
ax = axes(xlim=(0,Steps), ylim=(0,0.005))
ax.set_xlabel('Итерация'); ax.set_ylabel('J')
ax.plot(linspace(0,Steps-1,Steps),UU, color='black', ls='-', lw=2)
plt.grid(True)
print(UU)
