import time
import tqdm
import pandas as pd
import itertools
import polar_codes as pc
import numpy as np

df = pd.read_csv('rank.csv', sep=' ')

def run_simulation(null_message=True, N_list=[], R_list=[], L_list=[], SNR_range=[]):

    MAX_ITERS = 4000
    results = []
    total_iterations = len(L_list) * len(R_list) * len(N_list) * len(SNR_range) 

    with tqdm(total=total_iterations * MAX_ITERS, dynamic_ncols=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed} {rate_fmt} {postfix}') as pbar:
        start_time = time.time()
        for N, R, L, SNR in tqdm(itertools.product(N_list, R_list, L_list, SNR_range)):

            curr_failure = 0
            curr_iteration = 0

            while curr_iteration < MAX_ITERS:
                curr_iteration += 1
                        
                percent = pbar.n / (total_iterations * MAX_ITERS) * 100
                elapsed_time = pbar.format_interval(time.time() - start_time) 
                details = f'N={N}, R={R:.4f}, L={L}, SNR={SNR:.2f} dB, iters: {curr_iteration}, fails: {curr_failure}'
                pbar.set_postfix({'Progress': f'{percent:.2f}%', 'Time': elapsed_time, 'Details': details})

                        
                u, message_bits, info_indexes, frozen_indexes = pc.create_message(N, R, df, null_message)
                encoded_u = pc.polar_encode(u)
                encoded_u_bpsk = pc.bpsk(encoded_u)
                u_after_awgn, llrs = pc.AWGN(encoded_u_bpsk, SNR)

                decoded_u = pc.polar_decode_SCL(llrs, frozen_indexes, L)

                curr_failure = curr_failure + (0 if np.array_equal(u, decoded_u) else 1)

                pbar.update(1)


            results.append({
                'N': N,
                'R': f'{R:.2f}',
                'L': L,
                'SNR_dB': SNR,
                'FER': curr_failure / curr_iteration,
            })

    results_df = pd.DataFrame(results)
    return results_df