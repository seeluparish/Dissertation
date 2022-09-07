#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Created on Wed Sep  7 11:25:20 2022]

@author: pari
"""

#loading libraries 
import pandas as pd 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.metrics import mean_squared_error
import seaborn as sns

#Loading dataframe
with open('rawdataPOSCORAD.csv') as f:
    df = pd.read_csv(f, index_col = False)
f.close()

for col in df.columns:
    print(col)

df = df.reset_index(drop = True)

r_int_extent = 12.25
r_int_thickening = 0.027
r_int_swelling = 0.102
r_int_scratching = 0.051
r_int_redness = 0.073
r_int_oozing = 0.073
r_int_dryness = 0.102
r_int_sleep = 1
r_int_itching = 0.36

q_int_extent = 4
q_int_thickening = 0.008
q_int_swelling = 0.003
q_int_scratching = 0.032 
q_int_redness = 0.032
q_int_oozing = 0.044
q_int_dryness = 0.018
q_int_sleep = 0.423
q_int_itching = 0.25

def kalman_filter_extent(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    p = int(p)
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_extent = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_extent, x_list

def kalman_filter_thickening(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    p = int(p)
    
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_thickening = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_thickening, x_list

def kalman_filter_swelling(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    p = int(p)
    
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_swelling = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_swelling , x_list

def kalman_filter_scratching(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    p = int(p)
    
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_scratching = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_scratching, x_list

def kalman_filter_redness(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    p = int(p)
    
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_redness = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_redness, x_list

def kalman_filter_oozing(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    p = int(p)
    
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_oozing = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_oozing, x_list

def kalman_filter_dryness(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    p = int(p)
    
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_dryness = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_dryness, x_list

def kalman_filter_sleep(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    p = int(p)
    
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_sleep = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_sleep, x_list

def kalman_filter_itching(p, q, r, Z):
    # n: number of iterations to run the filter for
    # dt: time interval for updates
    # v: velocity of robot
    # p_v: uncertainty in velocity
    # q: process noise variance (uncertainty in the system's dynamic model)
    # r: measurement uncertainty
    # Z: list of position estimates derived from sensor measurements
    # Initialize state (x) and state uncertainty (p)
    
    x_list = []
    p_list = []
    
    p_v = 1
    v = 1
    dt = 0.25
    x = Z[0]
    
    
    
    n = len(Z)

    for i in range(n):
        # 1. Prediction
        # Predict the state using the system's dynamic model
        x = x + dt*v 
        p = p + ((dt**2) * p_v) + q 

        # 2. Measure
        # Get sensor measurement
        if Z[i] == 10000:
            z = x
        else:
            z = Z[i]

        # 3. Update
        # Compute the Kalman Gain
        k = p / ( p + r) 
    # Update the state and state uncertainty
        x = x + k * (z - x)
        p = (1 - k) * p

        x_list.append(x)
        p_list.append(p)
        
        p_update_itching = p_list[-1]
        
    # Return the final state and uncertainty
    return p_update_itching, x_list

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


#extent
p_update = 5
patient = df[df.Patient == 1]
patient = patient.fillna(10000)
z_extent = patient['Extent'].to_list()

z1 = []

for i in range(len(z_extent)):
    z1.append(int(z_extent[i]))
Z_extent = z1

p_update,x_extent = kalman_filter_extent(p_update, q_int_extent, r_int_extent, Z_extent)



for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)

    # Extent 
    z_extent = patient['Extent'].to_list()

    z1 = []

    for i in range(len(z_extent)):
        z1.append(int(z_extent[i]))
    Z_extent = z1
    
    p_update, x_extent = kalman_filter_extent(p_update, q_int_extent, r_int_extent, Z_extent)

rms_extent_list = []
p325_x_extent = []
p325_Z_extent = []

p331_x_extent = []
p331_Z_extent = []

for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)
    
    z_extent = patient['Extent'].to_list()

    z1 = []

    for i in range(len(z_extent)):
        z1.append(int(z_extent[i]))
    Z_extent = z1

    p_update, x_extent = kalman_filter_extent(p_update, q_int_extent, r_int_extent, Z_extent)
    
    
    rms_extent = mean_squared_error(x_extent, Z_extent, squared=False)
    rms_extent_list.append(rms_extent)
    
    if j == 325:
        p325_x_extent = x_extent
        p325_Z_extent = Z_extent
        
    if j == 331:
        p331_x_extent = x_extent
        p331_Z_extent = Z_extent
        
    

print(p325_x_extent)
print(p325_Z_extent)


for i in rms_extent_list:
    if i > 10:
        print(print)
        rms_extent_list.remove(i)
        
# dryness
p_update_dryness = 5
patient = df[df.Patient == 1]
patient = patient.fillna(10000)
z_dryness = patient['Dryness'].to_list()
z2 = []



for i in range(len(z_dryness)):
    z2.append(int(z_dryness[i]))
Z_dryness = z2
p_update_dryness, x_dryness = kalman_filter_dryness(p_update_dryness, q_int_dryness, r_int_dryness, Z_dryness)


for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)
    
    z_dryness = patient['Dryness'].to_list()
    z2 = []

    for i in range(len(z_dryness)):
        z2.append(int(z_dryness[i]))
    Z_dryness = z2
    
    p_update_dryness, x_dryness = kalman_filter_dryness(p_update_dryness, q_int_dryness, r_int_dryness, Z_dryness)
print(p_update_dryness)

rms_dryness_list = []

p325_x_dryness = []
p325_Z_dryness = []

p331_x_dryness = []
p331_Z_dryness = []

for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)
    
    z_dryness = patient['Dryness'].to_list()
    z2 = []

    for i in range(len(z_dryness)):
        z2.append(int(z_dryness[i]))
    Z_dryness = z2
    
    p_update_dryness, x_dryness = kalman_filter_dryness(p_update_dryness, q_int_dryness, r_int_dryness, Z_dryness)
    
    rms_dryness = mean_squared_error(x_dryness, Z_dryness, squared=False)
    rms_dryness_list.append(rms_dryness)
    
    if j == 325:
        p325_x_dryness = x_dryness
        p325_Z_dryness = Z_dryness
        
    if j == 331:
        p331_x_dryness = x_dryness
        p331_Z_dryness = Z_dryness

for i in rms_dryness_list:
    if i > 10:
        print(print)
        rms_dryness_list.remove(i)
        
print(rms_dryness_list)

# Thickening

p_update = 0.5

patient = df[df.Patient == 1]
patient = patient.fillna(10000)
    
# swelling
z_thickening = patient['Thickening'].to_list()
z4 = []

for i in range(len(z_thickening)):
    z4.append(int(z_thickening[i]))
Z_thickening = z4
p_update_thickening, x_thickening = kalman_filter_thickening(p_update, q_int_thickening, r_int_thickening, Z_thickening)


for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)
    
    z_thickening = patient['Thickening'].to_list()
    z3 = []

    for i in range(len(z_thickening)):
                   z3.append(int(z_thickening[i]))              
    Z_thickening = z3
    
    p_update_tickening, x_thickening = kalman_filter_thickening(p_update_thickening, q_int_thickening, r_int_thickening, Z_thickening)
    

rms_thickening_list = []

p325_x_thickening = []
p325_Z_thickening = []

p331_x_thickening = []
p331_Z_thickening = []

for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)
    
    z_thickening = patient['Thickening'].to_list()
    z3 = []

    for i in range(len(z_thickening)):
                   z3.append(int(z_thickening[i]))              
    Z_thickening = z3
    
    p_update_tickening, x_thickening = kalman_filter_thickening(p_update_thickening, q_int_thickening, r_int_thickening, Z_thickening)
    
    rms_thickening = mean_squared_error(x_thickening, Z_thickening, squared=False)
    rms_thickening_list.append(rms_thickening)
    
    if j == 325:
        p325_x_thickening = x_thickening
        p325_Z_thickening = Z_thickening
        
    if j == 331:
        p331_x_thickening = x_thickening
        p331_Z_thickening = Z_thickening

for i in rms_thickening_list:
    if i > 1:
        print(print)
        rms_thickening_list.remove(i)
        
# Swelling
p_update = 0.5

patient = df[df.Patient == 1]
patient = patient.fillna(10000)
    
# swelling
z_swelling = patient['Swelling'].to_list()
z4 = []

for i in range(len(z_swelling)):
    z4.append(int(z_swelling[i]))
Z_swelling = z4
p_update_swelling, x_swelling = kalman_filter_swelling(p_update, q_int_swelling, r_int_swelling, Z_swelling)


for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)
    
    # swelling
    z_swelling = patient['Swelling'].to_list()
    z4 = []

    for i in range(len(z_swelling)):
        z4.append(int(z_swelling[i]))
    Z_swelling = z4

    p_update_swelling, x_swelling = kalman_filter_swelling(p_update_swelling, q_int_swelling, r_int_swelling, Z_swelling)
print(p_update_swelling)

rms_swelling_list = []
p325_x_swelling = []
p325_Z_swelling = []

p331_x_swelling = []
p331_Z_swelling = []

for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)
    
    # swelling
    z_swelling = patient['Swelling'].to_list()
    z4 = []

    for i in range(len(z_swelling)):
        z4.append(int(z_swelling[i]))
    Z_swelling = z4

    p_update_swelling, x_swelling = kalman_filter_swelling(p_update_swelling, q_int_swelling, r_int_swelling, Z_swelling)

    rms_swelling = mean_squared_error(x_swelling, Z_swelling, squared=False)
    rms_swelling_list.append(rms_swelling)
    
    if j == 325:
        p325_x_swelling = x_swelling
        p325_Z_swelling = Z_swelling
        
    if j == 331:
        p331_x_swelling = x_swelling
        p331_Z_swelling = Z_swelling
    
for i in rms_swelling_list:
    if i > 1:
        print(print)
        rms_swelling_list.remove(i)
        
# Scratching
p_update = 0.5
patient = df[df.Patient == 1]
patient = patient.fillna(10000)
    
        # Scratching 
z_scratching = patient['Scratching'].to_list()
z6 = []

for i in range(len(z_scratching)):
    z6.append(int(z_scratching[i]))
Z_scratching = z6

p_update_scratching, x_scratching = kalman_filter_scratching(p_update, q_int_scratching, r_int_scratching, Z_scratching)


for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)
    
        # Scratching 
    z_scratching = patient['Scratching'].to_list()
    z6 = []

    for i in range(len(z_scratching)):
        z6.append(int(z_scratching[i]))
    Z_scratching = z6
    
    p_update_scratching, x_scratching = kalman_filter_scratching(p_update_scratching, q_int_scratching, r_int_scratching, Z_scratching)
print(p_update_scratching)

rms_scratching_list = []

p325_x_scratching = []
p325_Z_scratching = []

p331_x_scratching = []
p331_Z_scratching = []

for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)
    
        # Scratching 
    z_scratching = patient['Scratching'].to_list()
    z6 = []

    for i in range(len(z_scratching)):
        z6.append(int(z_scratching[i]))
    Z_scratching = z6
    
    p_update_scratching, x_scratching = kalman_filter_scratching(p_update_scratching, q_int_scratching, r_int_scratching, Z_scratching)
    rms_scratching = mean_squared_error(x_scratching, Z_scratching, squared=False)
    rms_scratching_list.append(rms_scratching)
    
    if j == 325:
        p325_x_scratching = x_scratching
        p325_Z_scratching = Z_scratching
        
    if j == 331:
        p331_x_scratching = x_scratching
        p331_Z_scratching = Z_scratching
    
for i in rms_scratching_list:
    if i > 1:
        print(print)
        rms_scratching_list.remove(i)
        
p_update = 0.5
patient = df[df.Patient == 1]
patient = patient.fillna(10000)
    
z_redness = patient['Redness'].to_list()
z6 = []

for i in range(len(z_redness)):
    z6.append(int(z_redness[i]))
Z_redness = z6

p_update_redness, x_redness = kalman_filter_redness(p_update, q_int_redness, r_int_redness, Z_redness)

for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)

    # redness
    z_redness = patient['Redness'].to_list()
    z3 = []

    for i in range(len(z_redness)):
        z3.append(int(z_redness[i]))
    Z_redness = z3
    
    p_update_redness, x_redness = kalman_filter_redness(p_update_redness, q_int_redness, r_int_redness, Z_redness)
    
print(p_update_redness)

rms_redness_list = []
p325_x_redness = []
p325_Z_redness = []

p331_x_redness = []
p331_Z_redness = []
    
for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)

    # redness
    z_redness = patient['Redness'].to_list()
    z3 = []

    for i in range(len(z_redness)):
        z3.append(int(z_redness[i]))
    Z_redness = z3
    
    p_update_redness, x_redness = kalman_filter_redness(p_update_redness, q_int_redness, r_int_redness, Z_redness)
    rms_redness = mean_squared_error(x_redness, Z_redness, squared=False)
    rms_redness_list.append(rms_redness)
    
    if j == 325:
        p325_x_redness = x_redness
        p325_Z_redness = Z_redness
        
    if j == 331:
        p331_x_redness = x_redness
        p331_Z_redness = Z_redness
    
for i in rms_redness_list:
    if i > 1:
        print(print)
        rms_redness_list.remove(i)

# Itching
p_update = 7
patient = df[df.Patient == 1]
patient = patient.fillna(10000)

# itching 
z_itching = patient['Itching VAS'].to_list()
z8 = []

for i in range(len(z_itching)):
    z8.append(int(z_itching[i]))
Z_itching = z8

p_update_itching, x_itching = kalman_filter_itching(p_update, q_int_itching, r_int_itching, Z_itching)

for k in range(2,301):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)
    
        
    # itching 
    z_itching = patient['Itching VAS'].to_list()
    z8 = []

    for i in range(len(z_itching)):
        z8.append(int(z_itching[i]))
    Z_itching = z8
    p_update_itching, x_itching = kalman_filter_itching(p_update_itching, q_int_itching, r_int_itching, Z_itching)
print(p_update_itching)

rms_itching_list = []
p325_x_extent = []
p325_Z_extent = []

p331_x_extent = []
p331_Z_extent = []


for j in range(330, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)
    
        
    # itching 
    z_itching = patient['Itching VAS'].to_list()
    z8 = []

    for i in range(len(z_itching)):
        z8.append(int(z_itching[i]))
    Z_itching = z8
    p_update_itching, x_itching = kalman_filter_itching(p_update_itching, q_int_itching, r_int_itching, Z_itching)
    rms_itching = mean_squared_error(x_itching, Z_itching, squared=False)
    rms_itching_list.append(rms_itching)
    
for i in rms_itching_list:
    if i > 1:
        print(print)
        rms_itching_list.remove(i)
print(rms_itching_list)

# oozing
p_update = 0.5
patient = df[df.Patient == 1]
patient = patient.fillna(10000)
    
z_oozing = patient['Oozing'].to_list()
z5 = []

for i in range(len(z_oozing)):
    z5.append(int(z_oozing[i]))
Z_oozing = z5
p_update_oozing, x_oozing= kalman_filter_oozing(p_update, q_int_oozing, r_int_oozing, Z_oozing)



for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)
    
    a = []
    # oozing 
    z_oozing = patient['Oozing'].to_list()
    z5 = []

    for i in range(len(z_oozing)):
        z5.append(int(z_oozing[i]))
    Z_oozing = z5
    p_update_oozing, x_oozing = kalman_filter_oozing(p_update_oozing, q_int_oozing, r_int_oozing, Z_oozing)
print(p_update_oozing)

rms_oozing_list = []
p325_x_oozing = []
p325_Z_oozing = []

p331_x_oozing = []
p331_Z_oozing = []

for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)
    
    # oozing 
    z_oozing = patient['Oozing'].to_list()
    z5 = []

    for i in range(len(z_oozing)):
        z5.append(int(z_oozing[i]))
    Z_oozing = z5
    p_update_oozing, x_oozing = kalman_filter_oozing(p_update_oozing, q_int_oozing, r_int_oozing, Z_oozing)
    rms_oozing = mean_squared_error(x_oozing, Z_oozing, squared=False)
    rms_oozing_list.append(rms_oozing)
    
    if j == 325:
        p325_x_oozing = x_oozing
        p325_Z_oozing = Z_oozing
        
    if j == 331:
        p331_x_oozing = x_oozing
        p331_Z_oozing = Z_oozing
    
for i in rms_oozing_list:
    if i > 1:
        print(i)
        rms_oozing_list.remove(i)
        
    rms_oozing_list.remove(rms_oozing_list[7])
        
print(rms_oozing_list)

#Sleep

p_update = 7
patient = df[df.Patient == 1]
patient = patient.fillna(10000)

    # sleep disturbance

z_sleep = patient['Sleep disturbance VAS'].to_list()
z9 = []

for i in range(len(z_sleep)):
    z9.append(int(z_sleep[i]))
Z_sleep = z9

p_update_sleep, x_sleep = kalman_filter_sleep(p_update, q_int_sleep, r_int_sleep, Z_sleep)



for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)

    # sleep disturbance

    z_sleep = patient['Sleep disturbance VAS'].to_list()
    z9 = []

    for i in range(len(z_sleep)):
        z9.append(int(z_sleep[i]))
    Z_sleep = z9
    p_update_sleep, x_sleep = kalman_filter_sleep(p_update_sleep, q_int_sleep, r_int_sleep, Z_sleep)
print(p_update_sleep)

rms_sleep_list = []
p325_x_sleep = []
p325_Z_sleep = []

p331_x_sleep = []
p331_Z_sleep = []


for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)

    # sleep disturbance

    z_sleep = patient['Sleep disturbance VAS'].to_list()
    z9 = []

    for i in range(len(z_sleep)):
        z9.append(int(z_sleep[i]))
    Z_sleep = z9
    p_update_sleep, x_sleep = kalman_filter_sleep(p_update_sleep, q_int_sleep, r_int_sleep, Z_sleep)
    rms_sleep = mean_squared_error(x_sleep, Z_sleep, squared=False)
    rms_sleep_list.append(rms_sleep)
    
    if j == 325:
        p325_x_sleep = x_sleep
        p325_Z_sleep = Z_sleep
        
    if j == 331:
        p331_x_sleep = x_sleep
        p331_Z_sleep = Z_sleep
    
    
for i in rms_sleep_list:
    if i > 1:
        print(print)
        rms_sleep_list.remove(i)
print(rms_sleep_list)

# Itching

p_update = 7
patient = df[df.Patient == 1]
patient = patient.fillna(10000)


z_itching = patient['Itching VAS'].to_list()
z9 = []

for i in range(len(z_itching)):
    z9.append(int(z_itching[i]))
Z_itching = z9

p_update_itching, x_itching = kalman_filter_itching(p_update, q_int_itching, r_int_itching, Z_itching)



for k in range(2,313):
    patient = df[df.Patient == k]
    patient = patient.fillna(10000)

    # sleep disturbance

    z_itching = patient['Itching VAS'].to_list()
    z9 = []

    for i in range(len(z_itching)):
        z9.append(int(z_itching[i]))
    Z_itching = z9
    p_update_itching, x_itching = kalman_filter_itching(p_update_itching, q_int_itching, r_int_itching, Z_itching)
print(p_update_itching)

rms_itching_list = []
p325_x_extent = []
p325_Z_extent = []

p331_x_extent = []
p331_Z_extent = []

for j in range(313, 347):
    patient = df[df.Patient == j]
    patient = patient.fillna(10000)

    # sleep disturbance

    z_itching= patient['Itching VAS'].to_list()
    z9 = []

    for i in range(len(z_itching)):
        z9.append(int(z_itching[i]))
    Z_itching = z9
    p_update_itching, x_itching= kalman_filter_itching(p_update_itching, q_int_itching, r_int_itching, Z_itching)
    rms_itching= mean_squared_error(x_itching, Z_itching, squared=False)
    rms_itching_list.append(rms_itching)
    
    if j == 325:
        p325_x_itching = x_itching
        p325_Z_itching = Z_itching
        
    if j == 331:
        p331_x_itching = x_itching
        p331_Z_itching = Z_itching
    
for i in rms_itching_list:
    if i > 1:
        print(print)
        rms_itching_list.remove(i)
print(rms_itching_list)


patient = df[df.Patient == 9]
patient = patient.fillna(10000)
    
# swelling
z_swelling = patient['Itching VAS'].to_list()
z4 = []
for i in range(len(z_swelling)):
    z4.append(int(z_swelling[i]))
Z_swelling = z4

print(Z_swelling)

x = np.linspace(0,len(Z_swelling), num = len(Z_swelling))
y = np.array(Z_swelling)

plt.plot(x, y)
plt.xlabel("Time")
plt.ylabel("Itching VAS")

plt.xticks([])
plt.ylim(-0.1,10)

plt.grid()
plt.show()