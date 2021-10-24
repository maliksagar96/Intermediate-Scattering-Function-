#Calculating intermediate scattering function 

import time
start=time.time()
import numpy as np
import pandas as pd
from pandas import DataFrame
import trackpy as tp
import matplotlib.pyplot as plt
import warnings
from warnings import warn

a=np.load('/home/sagar/Documents/codes/finalCodes/Cor057/linking/link057_PS1.npy')
traj=DataFrame(a,columns=['x','y','frame','particle'])

d=tp.compute_drift(traj)
tm=tp.subtract_drift(traj,d)
#print(tm)

traj=tm

#traj=traj[traj['particle']<=3]

#k=6.28
#k=2.71
k = 0.5
minus_one=-1

j=np.sqrt(np.complex(minus_one))
print(j)

#******begining of main code

def isf(traj, mpp, fps, max_lagtime=20000, pos_columns=None):   
    if pos_columns is None:
        pos_columns = ['x', 'y']
    result_columns = ['isf_real']

    # The following fails if > 1 record per particle (cannot reindex):
    try:
        # Reindex with consecutive frames, placing NaNs in the gaps.
        pos = traj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    except ValueError:
        if traj['frame'].nunique()!=len(traj['frame']):
            # Reindex failed due to duplicate index values
            raise Exception("Cannot use isf, more than one trajectory "
                            "per particle found.")
        else:
            raise



    max_lagtime = min(max_lagtime, len(pos) - 1)  # checking to be safe

    lagtimes = np.arange(1, max_lagtime + 1)

    result = pd.DataFrame(_isf_iter(pos.values, lagtimes),
                          columns=result_columns, index=lagtimes)
    #print(result)
    #result['delr'] = result[result_columns[-len(pos_columns):]].sum(1)
    #result['isf_real'] = np.real(np.exp(j*k*result['delr']))
    #result['isf_img']= np.imag(np.exp(j*k*result['delr']))
    result['lagt'] = result.index.values/float(fps)
    result.index.name = 'lagt'
    return result

def _isf_iter(pos, lagtimes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lt in lagtimes:
            diff = pos[lt:] - pos[:-lt]
            diff=np.real(np.exp((diff[:,0]+diff[:,1])*k*j))
            #print(np.shape(diff),type(diff))
            yield np.nanmean(diff, axis=0)




def eisf(traj, mpp, fps, max_lagtime=20000, pos_columns=None):
    ids = []
    isfs = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('particle'):
        print('pid', pid)
        #print('ptraj',ptraj)
        isfs.append(isf(ptraj, mpp, fps, max_lagtime, pos_columns))
        ids.append(pid)
    isfs = pd.concat(isfs, keys=ids, names=['particle', 'frame'])
    #print(isfs)   
    results=isfs.mean(level=1)
    #results['isf']=(results['isf_real']**2+results['isf_img']**2)**0.5
    #results['isf']=results['isf_real']
    #print(results)
    
    return results.set_index('lagt')['isf_real']


#*************end of main code



#result=isf(traj,145/512,1.92)
results=eisf(traj,145/512,21)
#print(results)

np.save('eisf_10k_0.76_average',results)
np.save('eisf_10k_0.76_average_index',results.index)

end=time.time()

print('Runtime : ',end-start)

# plt.title('EISF')
# plt.xscale('log')
# #plt.yscale('log')	
# #plt.axis('equal')
# plt.ylabel(r'$\langle \exp(i*k*\Delta r) \rangle$ [$\mu$m$^2$]')
# plt.xlabel('lag time $t$')
# plt.plot(results.index,results)
# plt.show()