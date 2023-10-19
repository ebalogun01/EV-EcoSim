"""This script is used post simulation to parse the voltages into cleaner voltages.csv"""

from voltdump2 import parse_voltages


month = 'January'
start_idx = 20
end_idx = 21
for i in range(start_idx, end_idx):
    folder_prefix = f'oneshot_{month}{i}/'
    parse_voltages(folder_prefix)
