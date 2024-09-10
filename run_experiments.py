import subprocess
from os import listdir

import os
Direc = "instances"
files = os.listdir(Direc)
# Filtering only the files.
names = [f[0:-5] for f in files if os.path.isfile(Direc+'/'+f)]
#print(*names, sep="\n")
names = sorted(names, reverse=True)
for name in names:
    subprocess.run(['python', 'main.py', name])
