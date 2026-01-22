from environment import run_simulation
from environment import plot_data
import pandas as pd
import numpy as np

N_list=[8, 32] # You can write your values
R_list=[1/3, 1/2] # You can write your values (no more than 4)
L_list=[1, 8] # You can write your values (no more than 4)

SNR_range=np.linspace(-10, 3, 10) # Your can write your values
data = run_simulation(N_list, R_list, L_list, SNR_range)

data.to_csv('data_test.csv', index=False)
data = pd.read_csv('data_test.csv')

plot_data(data)
