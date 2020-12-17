
from common.evaluate import evaluate
from models.unet.train_unet import train_unet
from models.unet.run_unet import run_unet
import shutil
import time
import pathlib
DEVICE=1
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(DEVICE)

start_time = time.perf_counter()
#remove old reconstruction files
dirpath = pathlib.Path(f'{DEVICE}rec_without')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
dirpath = pathlib.Path(f'{DEVICE}rec_with')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

acc=[8]
cf=[0.08]

#train_unet(False,acc,cf,DEVICE)
#run_unet(False,acc,cf,DEVICE)
#no_learning=evaluate(False,DEVICE)

train_unet(True,acc,cf,DEVICE)
run_unet(True,acc,cf,DEVICE)
with_learning=evaluate(True,DEVICE)

tt=time.perf_counter()-start_time
tth=tt/3600
print('****************************************************')
print(f'Device: {DEVICE}')
print(f'Total time: {tth}')
print('****************************************************')
print(f'acceleration {acc[0]},center of {cf[0]}, Regular:')
#print(no_learning)
print('With learning:')
print(with_learning)
print('****************************************************')