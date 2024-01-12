# this is used to pre-process the data from speech
import os

import pandas as pd
import numpy as np
import os

#%%
print(os.getcwd())
resolution = 1     # minutes
num_days = 1
num_nano_secs = int(num_days * 24 * 60 * 60 * 1e9)
step = int(resolution * 60 * 1e9)
dt_idx = pd.DatetimeIndex(list(range(0, num_nano_secs, step)), tz="UTC", freq=f'{resolution}T')

#%%
# todo: THIS TAKES IN THE 1MIN SPEECH DATA AND RESAMPLES IT TO 15 MIN INTERVALS FOR OPTIMIZATION
num_days = 30
num_evs = 3200
path_prefix = f'IndividualSessionsOutputData_{num_evs}/total_load_profiles_'
for day in range(num_days):
    in_data = pd.read_csv(f'{path_prefix}{day}.csv').set_index(dt_idx)  # Resampled from 1 min to 15 min with averaging
    if day == 0:
        out_data = in_data["Total"].resample('15T').mean().to_numpy()
    else:
        out_data = np.vstack((out_data,in_data["Total"].resample('15T').mean().to_numpy()))

np.savetxt(f'speechWeekdayLoad{num_evs}.csv', out_data)
print("max load is", out_data.max())

