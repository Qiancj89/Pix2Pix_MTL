import numpy as np
import shutil, os

state = ['TestData']
extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
for s in range(0, len(state)):        
    datapath = '../../'+state[s]+'/'

    if not os.path.exists('TestData/B_mode'):
        os.makedirs('TestData/B_mode')
    
    if not os.path.exists('TestData/CEUS'):
        os.makedirs('TestData/CEUS')
    
    if not os.path.exists('TestData/Mask'):
        os.makedirs('TestData/Mask')
    
    patient_name = os.listdir(datapath)

    for i in range(0, len(patient_name)):
        for ext in extensions:
            image_file = 'ZY_right' + ext
            if os.path.exists(datapath+patient_name[i]+'/Original/'+image_file):
                print(datapath+patient_name[i]+'/Original/'+image_file)
                shutil.copy(datapath+patient_name[i]+'/Original/'+image_file, 'TestData/B_mode/'+patient_name[i]+'.jpg')
            else:
                continue
        
        for ext in extensions:
            image_file = 'ZY_left' + ext
            if os.path.exists(datapath+patient_name[i]+'/Original/'+image_file):
                print(datapath+patient_name[i]+'/Original/'+image_file)
                shutil.copy(datapath+patient_name[i]+'/Original/'+image_file, 'TestData/CEUS/'+patient_name[i]+'.jpg')
            else:
                continue
        
        for ext in extensions:
            image_file = 'ZY_left' + ext
            if os.path.exists(datapath+patient_name[i]+'/Mask/'+image_file):
                print(datapath+patient_name[i]+'/Mask/'+image_file)
                shutil.copy(datapath+patient_name[i]+'/Mask/'+image_file, 'TestData/Mask/'+patient_name[i]+'.jpg')
            else:
                continue