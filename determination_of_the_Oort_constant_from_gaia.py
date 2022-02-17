import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Loading data from csv file | csv -> (comma-separated-values)
df = pd.read_csv('dados_gaia.csv', encoding='ISO-8859-1')

# Calculating distances and adding to dataFrame | Parallax is a difference in the apparent position of an object viewed along two different lines of sight
df['range'] = 1000/(df['parallax']) # The range unit is 'pc'

# We need to select a sample of star in the vicinity of the solar system | Then I will select the distant stars up to 100 pc from us
new_df = df.loc[df['range']<100]

# Resetting the indexes to stay organized
new_df = new_df.reset_index()

# We need to change the coordinate system of the proper motion which is in equatorial coordinates and we will mod to galactic coordinates.

# This transformation can be found in the article: https://arxiv.org/pdf/1306.2945.pdf
alpha_G = np.radians(192.85948)
delta_G = np.radians(27.12825)

# Calculating rotation matrix constants
c1 = np.sin(delta_G)*np.cos(np.radians(new_df['dec'])) - np.cos(delta_G)*np.sin(np.radians(new_df['dec']))*np.cos(np.radians(new_df['ra']) - alpha_G)
c2 = np.cos(delta_G)*np.sin(np.radians(new_df['ra']) - alpha_G)
cosb = np.sqrt(c1**2 + c2**2)

# Adjusting the coordinate system
pm_l_estrela = 1/cosb * (c1*new_df['pmra'] + c2*new_df['pmdec'])
pm_b = 1/cosb * (-c2*new_df['pmra'] + c1*new_df['pmdec'])

# Traverse speed with units set
vt = pm_l_estrela*new_df['range']*4.74047/(np.cos(np.radians(new_df['b']))*1000)

# There are a lot of stars so we need to take a sample and average on important variables to fit analytic function to data

# First and second step in galactic coordinates
step = 7.2
l1 = 0
l2 = step

# Declaring samples arrays
lsample = []
vtsample = []

l = np.array(new_df['l'])

while l2 <= 360:
    cond = np.argwhere( (l>l1) & (l<l2) ).flatten()
    
    lsample.append(np.mean((new_df['l'])[cond]))
    vtsample.append(np.mean(vt[cond]))
    
    # Evolution
    l1 = l2
    l2 = l2 + step

# Define function
def func(l, u0, v0):
    return -(u0*np.sin(np.radians(l)) + v0*np.cos(np.radians(l)))   

# Fit
params, cov = curve_fit(func, lsample, vtsample)

u0   = params[0]
v0   = params[1]

# Evaluate func using fitted parameters
xfit = np.linspace(0,360,100)
yfit = func(xfit, u0, v0)

# Now I will repeat the last procedure for all data

# Calculating rotation matrix constants
alphaG = np.radians(192.85948)
deltaG = np.radians(27.12825)

c1 = np.sin(deltaG)*np.cos(np.radians(df['dec'])) - np.cos(deltaG)*np.sin(np.radians(df['dec']))*np.cos(np.radians(df['ra']) - alphaG)
c2 = np.cos(deltaG)*np.sin(np.radians(df['ra']) - alphaG)
cosb = np.sqrt(c1**2 + c2**2)

# Adjusting the coordinate system
pm_l_star = 1/cosb * (c1*df['pmra'] + c2*df['pmdec'])


# Calculating the real proper motion with units set
correction = 1000*(u0*np.sin(np.radians(df['l'])) + v0*np.cos(np.radians(df['l'])))/df['range']
mpl = pm_l_star*4.74047/np.cos(np.radians(df['b'])) + correction

# There are a lot of stars so we need to take a sample and average on important variables to fit analytic function to data

# First and second step in galactic coordinates
step = 7.2
l1 = 0
l2 = step

# Declaring samples arrays
lsample = []
mplsample = []
l = np.array(df['l'])

while l2 <= 360:
    cond = np.argwhere( (l>l1) & (l<l2) ).flatten()
    
    lsample.append(np.mean(l[cond]))
    mplsample.append(np.mean(mpl[cond]))
    
    # Evolution
    l1 = l2
    l2 = l2 + step

# Define function
def func(l, A, B):
    return A*np.cos(2*np.radians(l)) + B  

# Fit
params, cov = curve_fit(func, lsample, mplsample)

A   = params[0]
B   = params[1]

# Evaluate func using fitted parameters
xfit = np.linspace(0,360,100)
yfit = func(xfit, A, B)

# To look at the graph and the Oort constant A and B
plt.plot(lsample, mplsample, 'o', color='b', label='Gaia data')
plt.plot(xfit, yfit, color='r', label=f'$v_t(l) = {np.around(A, 2)}\ \cos(2l) {np.around(B, 2)}$')
plt.legend()

plt.xlabel('l ( °)')
plt.ylabel('$\mu_l$ (km s$^{-1}$ kpc$^{-1}$)')
plt.yticks(np.arange(-40, 40, 10))
plt.title('Proper Motion')
plt.savefig('proper-motion.jpg')