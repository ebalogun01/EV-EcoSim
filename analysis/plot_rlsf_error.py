import numpy as np
import matplotlib.pyplot as plt

import os


os.chdir('C:/Users/ebalo/Desktop/EV50_cosimulation/test_cases/rlsf/')
rmse_vmag=np.loadtxt('rmse_vmag.txt')

plt.figure(figsize=(10,6))
plt.plot(rmse_vmag)
plt.grid()
plt.ylabel('RMSE voltage magntide',fontsize=16)
plt.xlabel('Iteration (min)',fontsize=16)
plt.yscale('log')
ax=plt.gca()
ax.tick_params(axis='both', which='major', labelsize=16)

plt.show()
plt.savefig('rmse_vmag.png')