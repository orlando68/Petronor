# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:09:32 2018

@author: 106300
"""


import matplotlib.pyplot as plt
import numpy as np

# Data to plot
xx = np.linspace(0,100,50)
yy = (xx/10)**2

# Plot the data
ax1 = plt.subplot(1,1,1)
ax1.plot(xx,yy)
ax1.set_ylabel(r'ylabel')
ax1.set_xlabel(u'Temperature [\u2103]')

# Set scond x-axis
ax2 = ax1.twiny()

# Decide the ticklabel position in the new x-axis,
# then convert them to the position in the old x-axis
newlabel = [0,50,100,200] # labels of the xticklabels: the position in the new x-axis
k2degc = lambda t: t/2 # convert function: from Kelvin to Degree Celsius
newpos   = [k2degc(t) for t in newlabel]   # position of the xticklabels in the old x-axis
ax2.set_xticks(newpos)
ax2.set_xticklabels(newlabel)

ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
ax2.spines['bottom'].set_position(('outward', 36))
ax2.set_xlabel('Temperature [K]')
ax2.set_xlim(ax1.get_xlim())

# Save the figure
plt.savefig('two_xticks_under.png', bbox_inches='tight', pad_inches=0.02, dpi=150)