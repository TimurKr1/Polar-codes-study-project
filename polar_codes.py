import numpy as np
import pandas as pd
from typing import List, Tuple
from functools import lru_cache


def polar_encode(u: np.ndarray) -> np.ndarray:

    u = np.array(u, dtype=int)
    N = len(u)
    x = u.copy()
    n = 1

    while n < N:
        for i in range(0, N, 2 * n):
            x[i: i + n] ^= x[i + n: i + 2 * n]
        n *= 2

    return x


@lru_cache(maxsize=128)
def _cached_polar_encode(bits_tuple):

    return polar_encode(np.array(bits_tuple))


def L_step(llrs: np.ndarray) -> np.ndarray:

    N = len(llrs)
    x, y = llrs[ : N // 2], llrs[N // 2 : ] 

    return np.sign(x * y) * np.minimum(np.abs(x), np.abs(y))


def R_step(llrs: np.ndarray, bit: np.ndarray) -> np.ndarray:

    N = len(llrs)
    x, y = llrs[ : N // 2], llrs[N // 2 : ] 
    bit = np.array(bit) 

    return np.where(bit == 0, x + y, y - x)


def calc_llr_for_index(llrs: np.ndarray, decoded_bits: np.ndarray, index: int) -> float:

    N = len(llrs)

    if N == 1:
        return llrs[0]
    
    half = N // 2

    if index < half:
        return calc_llr_for_index(L_step(llrs), decoded_bits[:half], index)
    
    left_bits = decoded_bits[:half]
    u_hat = _cached_polar_encode(tuple(left_bits))

    return calc_llr_for_index(R_step(llrs, u_hat), decoded_bits[half:], index - half)


def extend_paths(paths: List[Tuple[List[int], float]], llr: float, frozen: bool) ->  List[Tuple[List[int], float]]:

    new_paths = []

    for bits, pm in paths:
        sc_bit = 0 if llr >= 0 else 1

        if frozen:
            new_pm = pm + abs(llr) * (0 != sc_bit)
            new_paths.append((bits + [0], new_pm))

        else:
            pm0 = pm + abs(llr) * (0 != sc_bit)
            new_paths.append((bits + [0], pm0))

            pm1 = pm + abs(llr) * (1 != sc_bit)
            new_paths.append((bits + [1], pm1))

    return new_paths


def cut_paths(paths:  List[Tuple[List[int], float]], L : int) -> List[Tuple[List[int], float]]:
    paths.sort(key=lambda p: p[1])
    return paths[:L]


def polar_decode_SC(llrs: np.ndarray, frozen_pos: np.ndarray) -> np.ndarray:
    N = len(llrs)
    decoded = []

    for i in range(N):
        llr_i = calc_llr_for_index(llrs, decoded, i)

        if i in frozen_pos:
            decoded.append(0)
        else:
            decoded.append(0 if llr_i >= 0 else 1)

    return np.array(decoded)


def polar_decode_SCL(llrs : np.ndarray, frozen_pos : np.ndarray, L : int) -> np.ndarray:
    N = len(llrs)
    paths = [([], 0.0)]

    for i in range(N):
        new_paths = []

        for bits, pm in paths:
            llr_i = calc_llr_for_index(llrs, bits, i)
            extended = extend_paths([(bits, pm)], llr_i, i in frozen_pos)
            new_paths.extend(extended)

        paths = cut_paths(new_paths, L)

    best_bits, _ = min(paths, key=lambda p: p[1])
    return np.array(best_bits)


def create_message(N : int, R: float, df: pd.DataFrame, null_message : bool):
    K = round(R * N)

    Q = df['Q'].values[df['Q'] < N]

    frozen_indexes = Q[:N-K].astype(int)
    info_indexes = Q[N-K:N].astype(int)

    message_bits = np.random.randint(0, 2, K) if not null_message else np.zeros(K)

    u = np.zeros(N, dtype=int)

    u[info_indexes] = message_bits

    return u, message_bits, info_indexes, frozen_indexes


def bpsk(x : np.ndarray):
    return np.where(x == 0, 1, -1)


def AWGN(x : np.ndarray, SNR_db: float):
    variance = 10 ** (-SNR_db / 10)

    noise = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=x.shape)

    y = x + noise
    llr = (2.0 * y) / variance

    return y, llr