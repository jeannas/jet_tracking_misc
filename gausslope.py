'''
Takes hdf5 files made from Silke's code and fits the Azimuthal average from CSPAD data to Gaussian on a slope.
'''

import matplotlib.pyplot as plt
import numpy as np
import math
from tables import open_file
from scipy.optimize import curve_fit

# open hdf5 file
# hdf5 file for cxilr6716, run 139
#dat = open_file('/reg/d/psdm/cxi/cxilr6716/results/smalldata_tools/output/cxilr6716_Run139.h5').root

# hdf5 file for cxilr8416, run 473
# this data appears to have two rings and does not fit well
dat = open_file('/reg/d/psdm/cxi/cxilr6716/results/smalldata_tools/output/cxilr8416_Run473.h5').root

# hdf5 file for cxilt5917, run 233
#dat = open_file('/reg/d/psdm/cxi/cxilr6716/results/smalldata_tools/output/cxilt5917_Run233.h5').root

# hdf5 file for cxilu1817, run 130
#dat = open_file('/reg/d/psdm/cxi/cxilr6716/results/smalldata_tools/output/cxilu1817_Run130.h5').root

# hdf5 file for cxilu5617, run 219
# split CSPAD - this data appears to have two rings but still fits relatively well
#dat = open_file('/reg/d/psdm/cxi/cxilr6716/results/smalldata_tools/output/cxilu5617_Run219.h5').root

# CSPAD for cxilr6716 run 139, cxilu5617 run 219
#total_events = dat.DsaCsPad.azav_azav_.shape[0]

# CSPAD for cxilr8416 run 473, cxilt5917 run 233, cxilu1817 run 130
total_events = dat.DscCsPad.azav_azav.shape[0]
print('total events: ', total_events)

# select event number
num = int(input('enter event number: '))

# ignore qbins that have less than 150 pixels
# CSPAD for cxilr6716 run 139, cxilu5617 run 219
#norm = dat.UserDataCfg.DsaCsPad.azav__azav_norm

# CSPAD for cxilr8416 run 473, cxilt5917 run 233, cxilu1817 run 130
norm = dat.UserDataCfg.DscCsPad.azav__azav_norm
start = 0
end = len(norm)
begin = math.trunc(end / 2)
for i in range(begin):
  a = begin - i
  b = begin + i
  if (norm[a] < 150) and (a > start):
    start = a
  if (norm[b] < 150) and (b < end):
    end = b

print(len(norm), start, end)

# extract Azimuthal average data for selected run from hdf5 file
# CSPAD for cxilr6716 run 139, cxilu5617 run 219
#azav = dat.DsaCsPad.azav_azav[num][0][start+1:end]

# CSPAD for cxilr8416 run 473, cxilt5917 run 233, cxilu1817 run 130
#azav = dat.DscCsPad.azav_azav[num][0][start+1:end]

# for cxilr8416 run 473, cxilt5917 run 233, cxilu1817 run 130, cxilu5617 run 219 there is Azimuthal average data using (0, 0) as center
#azav = dat.DsaCsPad.azav_c00_azav[num][0][start+1:end]
azav = dat.DscCsPad.azav_c00_azav[num][0][start+1:end]

x = np.arange(len(azav))

# estimate mean & standard deviation for Gaussian
n = len(x)
mean = sum(x*azav) / sum(azav)
std = np.sqrt(sum((x-mean)**2) / n)
#print(max(azav), mean, std)

# estimate slope for linear baseline
x0 = 50 / 2
l = len(azav)
x1 = l - (50/2)
y0 = np.mean(azav[0:50])
y1 = np.mean(azav[l-50:])
m, b = np.polyfit((x0, x1), (y0, y1), 1)
#print(m, b)

# define function for Gaussian + linear
def gaussianslope(x, a, mean, std, m, b):
    return (a * np.exp(-((x-mean)/2/std)**2)) + (m*x + b)

# fit Gaussian + linear to Azimuthal average data; provide initial parameters
popt, pcov = curve_fit(gaussianslope, x, azav, p0=[max(azav), mean, std, m, b])

# print optimized parameters 
print(popt)

# plot Azimuthal average data and fitted curve
plt.plot(x, azav, 'ro')
plt.plot((x0, x1), (y0, y1), 'k-')
plt.plot(x, gaussianslope(x, *popt), 'b-')
plt.show()
