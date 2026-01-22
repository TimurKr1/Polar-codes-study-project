import pandas as pd
import polar_codes as pc
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import seaborn as sns

df = pd.read_csv('rank.csv', sep=' ')


def run_simulation(N_list=[], R_list=[], L_list=[], SNR_range=[], null_message=True):

    MAX_ITERS = 4000
    results = []

    print(f"One combination has {MAX_ITERS} iterations.")

    for N, R, L, SNR in tqdm(itertools.product(N_list, R_list, L_list, SNR_range), 
                          desc="Processing combinations", 
                          total=len(N_list) * len(R_list) * len(L_list) * len(SNR_range), 
                          ncols=100, 
                          unit="combination",
                          colour='green'):

        curr_failure = 0
        curr_iteration = 0

        while curr_iteration < MAX_ITERS:
            curr_iteration += 1

            u, message_bits, info_indexes, frozen_indexes = pc.create_message(N, R, df, null_message)
            encoded_u = pc.polar_encode(u)
            encoded_u_bpsk = pc.bpsk(encoded_u)
            u_after_awgn, llrs = pc.AWGN(encoded_u_bpsk, SNR)

            decoded_u = pc.polar_decode_SCL(llrs, frozen_indexes, L)

            curr_failure = curr_failure + (0 if np.array_equal(u, decoded_u) else 1)


        results.append({
            'N': N,
            'R': f'{R:.2f}',
            'L': L,
            'SNR_dB': SNR,
            'FER': curr_failure / curr_iteration,
        })

    results_df = pd.DataFrame(results)
    return results_df


def plot_data(df: pd.DataFrame):

    unique_N = df['N'].unique()
    unique_R = df['R'].unique()
    unique_L = df['L'].unique()

    line_styles = ['-', '--', '-.', ':'] 
    markers = ['o', 's', '^', 'D']
    colors = sns.color_palette("Set1", n_colors=len(unique_N)) 

    R_line_styles = {R: line_styles[i % len(line_styles)] for i, R in enumerate(unique_R)}
    L_markers = {L: markers[i % len(markers)] for i, L in enumerate(unique_L)}

    plt.figure(figsize=(16, 10)) 

    for i, N in enumerate(unique_N):
        for j, R in enumerate(unique_R):
            for k, L in enumerate(unique_L):

                subset = df[(df['N'] == N) & (df['R'] == R) & (df['L'] == L)]

                plt.plot(subset['SNR_dB'], subset['FER'], 
                         label=f'N={N}, R={R}, L={L}', 
                         color=colors[i],  
                         linestyle=R_line_styles[R],  
                         marker=L_markers[L], 
                         markersize=4)

    plt.title('FER vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('FER')
    plt.yscale('log')
    plt.legend(title='N, R, L', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.show()