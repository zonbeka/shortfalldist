import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import matrix, asmatrix, kron, hstack, vstack, zeros
from itertools import product


def new_data(seed=None):
    # Set random generator
    rnd = random.Random()
    if seed is not None:
        rnd.seed(seed)
    result = {}
    # Generate random values for the demand
    phases = 2
    for target in ['Demand', 'Service']:
        initial = [rnd.random() for _ in range(phases)]
        total = sum(initial)
        for i in range(phases):
            initial[i] /= total
        transition = [[rnd.random() for _ in range(phases)] for _ in range(phases)]
        absorption = [[rnd.random() for _ in range(1)] for _ in range(phases)]
        for i in range(phases):
            total = sum(transition[i]) + absorption[i][0]
            for j in range(phases):
                transition[i][j] /= total
            absorption[i][0] /= total
        result[target] = {'Initial': initial,
                          'Transition': transition,
                          'Absorption': absorption
                          }
    return result


def crange(start, stop, step=1):
    d = 1 if (step > 0) else -1
    return range(start, stop + d, step)


def create_pmf(initial, t_matrix, absorption):
    pmf = [(initial * t_matrix ** (g - 1) * absorption)[0, 0] for g in range(1, 10)]
    pmf.insert(0, 0)
    cumulative = np.sum(pmf[g] for g in range(1, 10))
    assert 0 <= cumulative <= 1
    pmf[9] = pmf[9] + 1 - cumulative
    assert np.abs(np.sum(pmf[g] for g in range(1, 10)) - 1) <= 1e-10
    return pmf


def do_analysis(data_set):
    # Demand
    initial_demand = matrix(data_set['Demand']['Initial'])
    t_matrix_demand = matrix(data_set['Demand']['Transition'])
    absorption_demand = matrix(data_set['Demand']['Absorption'])
    # Service time
    initial_service = matrix(data_set['Service']['Initial'])
    t_matrix_service = matrix(data_set['Service']['Transition'])
    absorption_service = matrix(data_set['Service']['Absorption'])
    # Store results here
    result = {}

    pmf_demand = create_pmf(initial_demand, t_matrix_demand, absorption_demand)

    result['pmf_demand'] = {x: px for x, px in enumerate(pmf_demand) if x > 0}

    expected_demand = sum([d * pd for d, pd in enumerate(pmf_demand)])

    result['expected_demand'] = expected_demand

    result['sd_demand'] = np.sqrt(
        sum([d ** 2 * pd for d, pd in enumerate(pmf_demand)]) - expected_demand ** 2
    )

    pmf_service = create_pmf(initial_service, t_matrix_service, absorption_service)

    result['pmf_service'] = {x: px for x, px in enumerate(pmf_service) if x > 0}
    result['expected_service'] = sum([s * ps for s, ps in enumerate(pmf_service)])

    # Service time of batch
    initial_batch = kron(initial_demand, initial_service)
    t_matrix_batch = kron(np.identity(2), t_matrix_service) + kron(t_matrix_demand,
                                                                   absorption_service * initial_service)
    one = matrix([[1], [1], [1], [1]])
    absorption_batch = one - (t_matrix_batch * one)

    a_0 = absorption_batch * initial_batch
    a_1 = t_matrix_batch
    a_2 = kron(initial_batch, t_matrix_batch)
    a_3 = kron(t_matrix_batch, initial_batch)
    a_4 = kron(one, initial_batch)
    a_5 = kron(kron(absorption_batch, absorption_batch), initial_batch)
    a_6 = kron(t_matrix_batch, t_matrix_batch)
    a_7 = kron(t_matrix_batch, absorption_batch)
    a_8 = kron(absorption_batch, t_matrix_batch)
    a_9 = kron(t_matrix_batch, kron(absorption_batch, initial_batch))
    a10 = kron(kron(absorption_batch, initial_batch), t_matrix_batch)
    a11 = kron(kron(one, absorption_batch), initial_batch)
    a12 = kron(kron(one, initial_batch), t_matrix_batch)
    a13 = kron(kron(absorption_batch, one), initial_batch)
    a14 = kron(t_matrix_batch, kron(one, initial_batch))
    a15 = kron(kron(one, absorption_batch), kron(initial_batch, initial_batch))
    a16 = kron(kron(absorption_batch, one), kron(initial_batch, initial_batch))

    # Zero matrices
    u_1 = asmatrix(np.zeros((4, 16)))
    u_2 = asmatrix(np.zeros((16, 16)))
    u_3 = asmatrix(np.zeros((4, 4)))
    u_4 = asmatrix(np.zeros((16, 4)))

    transition_matrix = vstack((
        hstack((
            a_0, a_1, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, a_1, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, a_2, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, a_1, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_1, a_2, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, u_3, u_3, a_1, u_3, u_3, u_3, u_3, u_3, u_3, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_4, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, u_3, u_3, u_3, u_3, a_1, u_3, u_3, u_3, u_3, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_1, u_1, a_3, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, a_1, u_3, u_3, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_1, u_1, u_1, a_3, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_0, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, a_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_4, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_3, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1,
            u_1,
            u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1, u_1)),
        hstack((
            a_5, u_4, u_4, a_8, u_4, u_4, u_4, a_7, u_4, u_4, u_4, u_4, u_4, u_2, u_2, u_2, u_2, a_6, u_2, u_2, u_2,
            u_2,
            u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, u_4, u_4, u_4, u_4, a_8, u_4, a_7, u_4, u_4, u_4, u_4, u_4, u_2, u_2, u_2, u_2, u_2, a_6, u_2, u_2,
            u_2,
            u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, a_8, u_4, u_4, u_4, u_4, u_4, u_4, u_4, a_7, u_4, u_4, u_4, u_2, u_2, u_2, u_2, u_2, u_2, a_6, u_2,
            u_2,
            u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, a_8, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, a_7, u_4, u_2, u_2, u_2, u_2, u_2, u_2, u_2, a_6,
            u_2,
            u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, u_4, u_4, u_4, a_8, u_4, u_4, u_4, a_7, u_4, u_4, u_4, u_4, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            a_6,
            u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, u_4, u_4, u_4, u_4, u_4, a_8, u_4, a_7, u_4, u_4, u_4, u_4, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            u_2,
            a_6, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, u_4, a_8, u_4, u_4, u_4, u_4, u_4, u_4, u_4, a_7, u_4, u_4, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            u_2,
            u_2, a_6, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, u_4, a_8, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, a_7, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            u_2,
            u_2, u_2, a_6, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_2, a10, a_9, u_2, u_2, u_2, u_2, u_2,
            u_2,
            u_2, u_2, u_2, a_6, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a13, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_2, u_2, a14, u_2, u_2, u_2, u_2, u_2,
            u_2,
            u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            a_5, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, a10, u_2, u_2, a_9, u_2, u_2, u_2, u_2,
            u_2,
            u_2, u_2, u_2, u_2, a_6, u_2, u_2, u_2, u_2)),
        hstack((
            a11, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, a12, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            u_2,
            u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2)),
        hstack((
            u_4, a_5, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_2, u_2, u_2, u_2, u_2, a10, a_9, u_2,
            u_2,
            u_2, u_2, u_2, u_2, u_2, a_6, u_2, u_2, u_2)),
        hstack((
            u_4, a_5, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_2, u_2, u_2, u_2, a10, u_2, u_2, a_9,
            u_2,
            u_2, u_2, u_2, u_2, u_2, u_2, a_6, u_2, u_2)),
        hstack((
            u_4, u_4, a_5, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            u_2,
            a10, a_9, u_2, u_2, u_2, u_2, u_2, a_6, u_2)),
        hstack((
            u_4, u_4, a_5, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_2, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            a10,
            u_2, u_2, a_9, u_2, u_2, u_2, u_2, u_2, a_6)),
        hstack((
            u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, a16, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            u_2,
            u_2, u_2, u_2, u_2, a14, u_2, u_2, u_2, u_2)),
        hstack((
            u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, u_4, a15, u_2, u_2, u_2, u_2, u_2, u_2, u_2,
            u_2,
            u_2, u_2, u_2, a12, u_2, u_2, u_2, u_2, u_2))
    ))

    for i in range(0, 340):
        row_sum = 0
        for j in range(0, 340):
            row_sum = row_sum + transition_matrix[i, j]
        assert np.abs(row_sum - 1) < 1e-10

    # Calculate steady state probability
    pi = transition_matrix ** 500
    pi = np.asarray(np.transpose(pi[0, :]))

    # Collect all states in order from file
    states = {}
    with open('states.txt') as f:
        lines = f.readlines()
        f.close()
        for k, line in enumerate(lines):
            chars = list(line)
            i = int(chars[0])
            j = int(chars[1])
            x = int(chars[2])
            y = int(chars[3])
            assert 0 <= i <= 9 and 0 <= j <= 9 and 0 <= x <= 4 and 0 <= y <= 4
            states[(i, j, x, y)] = k

    # Transition probability matrix for the steady state
    pprr = zeros((10, 10, 10, 10))  # Transition probability
    p_state = zeros((10, 10))  # State probability
    for (i1, j1, x1, y1), k1 in states.items():
        p_state[i1][j1] += pi[k1]
        for (i2, j2, x2, y2), k2 in states.items():
            pprr[i1][j1][i2][j2] += pi[k1] * transition_matrix[k1, k2]

    # Lead time distribution
    p_lt = zeros(10)
    p_lt1 = zeros(10)
    p_lt2 = zeros(10)
    for i in crange(1, 9):
        for j in crange(0, 9):
            for t in crange(0, i):
                for k in crange(0, 9):
                    p_lt1[i] += pprr[i][j][t][k]
                    p_lt2[i] += pprr[j][i][k][t]
        p_lt[i] = p_lt1[i] + p_lt2[i]
    p_lt = p_lt / sum(p_lt)
    expected_lead_time_time_slots = sum([i * p_lt[i] for i in crange(0, 9)])
    p_ltp = zeros(4)
    p_ltp[1] = p_lt[1] + p_lt[2] + p_lt[3]
    p_ltp[2] = p_lt[4] + p_lt[5] + p_lt[6]
    p_ltp[3] = p_lt[7] + p_lt[8] + p_lt[9]
    expected_lead_time = sum([i * p_ltp[i] for i in crange(1, 3)])

    result['pmf_lead_time_in_time_slots'] = {x: px for x, px in enumerate(p_lt) if x > 0}
    result['pmf_lead_time_in_periods'] = {x: px for x, px in enumerate(p_ltp) if x > 0}
    result['expected_lead_time_in_time_slots'] = expected_lead_time_time_slots
    result['expected_lead_time_in_periods'] = expected_lead_time

    result['sd_lead_time_in_time_slots'] = np.sqrt(sum(
        [i ** 2 * p_lt[i] for i in crange(0, 9)]) - expected_lead_time_time_slots ** 2)

    # Wait time distribution
    state_subset1 = [(1, 0), (2, 0), (5, 0), (8, 0), (0, 2), (0, 5), (0, 8), (1, 4), (4, 1), (5, 2), (2, 5), (1, 7),
                     (7, 1),
                     (8, 2), (2, 8)]
    state_subset2 = [(3, 0), (0, 3), (6, 0), (0, 6), (9, 0), (0, 9), (3, 6), (6, 3), (3, 9), (9, 3), (6, 9), (9, 6)]
    arrival_state = sum([pprr[i][j][1][0] for i, j in state_subset1]) + sum([p_state[i][j] for i, j in state_subset2])
    p_w = zeros(4)
    # W = 1
    p_w[1] = pprr[3][6][4][7] * (
        sum([pprr[4][7][i][j] for i, j in product(crange(0, 9), repeat=2)]) - pprr[4][7][5][8]
    ) / p_state[4][7] / arrival_state
    p_w[1] += pprr[6][3][7][4] * (
        sum([pprr[7][4][i][j] for i, j in product(crange(0, 9), repeat=2)]) - pprr[7][4][8][5]
    ) / p_state[7][4] / arrival_state
    p_w[1] += pprr[6][9][7][4] * (
        sum([pprr[7][4][i][j] for i, j in product(crange(0, 9), repeat=2)]) - pprr[7][4][8][5]
    ) / p_state[7][4] / arrival_state
    p_w[1] += pprr[9][6][4][7] * (
        sum([pprr[4][7][i][j] for i, j in product(crange(0, 9), repeat=2)]) - pprr[4][7][5][8]
    ) / p_state[4][7] / arrival_state
    # W = 2
    p_w[2] = pprr[3][6][4][7] * pprr[4][7][5][8] * (
        sum([pprr[5][8][i][j] for i, j in product(crange(0, 9), repeat=2)]) - pprr[5][8][6][9]
    ) / p_state[4][7] / p_state[5][8] / arrival_state
    p_w[2] += pprr[6][3][7][4] * pprr[7][4][8][5] * (
        sum([pprr[8][5][i][j] for i, j in product(crange(0, 9), repeat=2)]) - pprr[8][5][9][6]
    ) / p_state[7][4] / p_state[8][5] / arrival_state
    p_w[2] += pprr[6][9][7][4] * pprr[7][4][8][5] * (
        sum([pprr[8][5][i][j] for i, j in product(crange(0, 9), repeat=2)]) - pprr[8][5][9][6]
    ) / p_state[7][4] / p_state[8][5] / arrival_state
    p_w[2] += pprr[9][6][4][7] * pprr[4][7][5][8] * (
        sum([pprr[5][8][i][j] for i, j in product(crange(0, 9), repeat=2)]) - pprr[5][8][6][9]
    ) / p_state[4][7] / p_state[5][8] / arrival_state
    # W = 3
    p_w[3] = pprr[3][6][4][7] * pprr[4][7][5][8] * pprr[5][8][6][9] / p_state[4][7] / p_state[5][8] / arrival_state
    p_w[3] += pprr[6][3][7][4] * pprr[7][4][8][5] * pprr[8][5][9][6] / p_state[7][4] / p_state[8][5] / arrival_state
    p_w[3] += pprr[6][9][7][4] * pprr[7][4][8][5] * pprr[8][5][9][6] / p_state[7][4] / p_state[8][5] / arrival_state
    p_w[3] += pprr[9][6][4][7] * pprr[4][7][5][8] * pprr[5][8][6][9] / p_state[4][7] / p_state[5][8] / arrival_state
    # W = 0
    p_w[0] = 1 - sum([p_w[i] for i in crange(1, 3)])
    # Collect result
    result['pmf_wait_time'] = {x: px for x, px in enumerate(p_w)}

    # Joint Distribution of Quantity and Service Time
    p_qt = zeros((10, 10))
    p_t_convoluted = pmf_service
    for q in crange(1, 9):
        for t in crange(1, 9):
            part1 = pmf_demand[q]
            part2 = p_t_convoluted[t]
            part3 = 0
            if t == 9:
                part3 = sum([p_t_convoluted[i] for i in range(len(p_t_convoluted)) if i > 9])
            p_qt[q][t] = part1 * (part2 + part3)
        p_t_convoluted = np.convolve(pmf_service, p_t_convoluted)

    # Shortfall distribution
    p_sf = zeros(9 * 3 + 1)
    for qo, qm, qn, so, sm, sn in product(crange(1, 9), repeat=6):
        if so == 1:
            if sm == 1:
                sum_q = qn
                p_sf[sum_q] += p_qt[qo][so] * p_qt[qm][sm] * p_qt[qn][sn]
            elif 2 <= sm <= 3:
                sum_q = qm + qn
                for wm in crange(0, sm - 1):
                    p_sf[sum_q] += p_qt[qo][so] * p_qt[qm][sm - wm] * p_w[wm] * p_qt[qn][sn]
            else:
                sum_q = qm + qn
                for wm in crange(0, 3):
                    p_sf[sum_q] += p_qt[qo][so] * p_qt[qm][sm - wm] * p_w[wm] * p_qt[qn][sn]
        elif 2 <= so <= 6:
            if sm == 1:
                sum_q = qn
                for wo in crange(0, min(so - 1, 3)):
                    p_sf[sum_q] += p_qt[qo][so - wo] * p_w[wo] * p_qt[qm][sm] * p_qt[qn][sn]
            elif 2 <= sm <= 3:
                sum_q = qn
                for wo in crange(0, min(so - 1, 3)):
                    for wm in crange(0, sm - 1):
                        p_sf[sum_q] += p_qt[qo][so - wo] * p_w[wo] * p_qt[qm][sm - wm] * p_w[wm] * p_qt[qn][sn]
            else:
                sum_q = qm + qn
                for wo in crange(0, min(so - 1, 3)):
                    for wm in crange(0, 3):
                        p_sf[sum_q] += p_qt[qo][so - wo] * p_w[wo] * p_qt[qm][sm - wm] * p_w[wm] * p_qt[qn][sn]
        else:
            if sm == 1:
                sum_q = qo + qn
                for wo in crange(0, 3):
                    p_sf[sum_q] += p_qt[qo][so - wo] * p_w[wo] * p_qt[qm][sm] * p_qt[qn][sn]
            elif 2 <= sm <= 3:
                sum_q = qo + qn
                for wo in crange(0, 3):
                    for wm in crange(0, sm - 1):
                        p_sf[sum_q] += p_qt[qo][so - wo] * p_w[wo] * p_qt[qm][sm - wm] * p_w[wm] * p_qt[qn][sn]
            else:
                sum_q = qo + qm + qn
                for wo in crange(0, 3):
                    for wm in crange(0, 3):
                        p_sf[sum_q] += p_qt[qo][so - wo] * p_w[wo] * p_qt[qm][sm - wm] * p_w[wm] * p_qt[qn][sn]

    result['pmf_shortfall'] = {x: px for x, px in enumerate(p_sf)}

    expected_shortfall = sum([sf * psf for sf, psf in enumerate(p_sf)])
    result['expected_shortfall'] = expected_shortfall
    result['sd_shortfall'] = np.sqrt(
        sum([sf ** 2 * psf for sf, psf in enumerate(p_sf)]) - expected_shortfall ** 2)

    expected_wait_time = sum([w * pw for w, pw in enumerate(p_w)])
    result['expected_wait_time'] = expected_wait_time
    result['sd_wait_time'] = np.sqrt(
        sum([w ** 2 * pw for w, pw in enumerate(p_w)]) - expected_wait_time ** 2)

    # Base Stock Level
    fill_rate = 0.95
    aana = (1 - fill_rate) * expected_demand
    zz = zeros(31)
    for base_stock in crange(1, 30):
        for shortfall in crange(1, 27):
            zz[base_stock] += p_sf[shortfall] * max(shortfall - base_stock, 0)
    min_base_stock = min([a for a, z in enumerate(zz) if z <= aana and a > 0])
    result['safety_stock'] = min_base_stock - (expected_lead_time + 1) * expected_demand

    # Return all results
    return result


def make_graphs(model_results):
    # Make plot
    plt.subplot(221)
    x = []
    y = []
    for k, v in model_results['pmf_lead_time_in_time_slots'].items():
        x.append(k)
        y.append(v)
    plt.bar(x, y)
    plt.xticks([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])
    plt.xlabel('Time slots')
    plt.ylabel('Probability')
    plt.title('Lead-time distribution in time slots')

    plt.subplot(222)
    x = []
    y = []
    for k, v in model_results['pmf_lead_time_in_periods'].items():
        x.append(k)
        y.append(v)
    plt.bar(x, y)
    plt.xticks([1, 2, 3], [1, 2, 3])
    plt.xlabel('Time periods')
    plt.ylabel('Probability')
    plt.title('Lead-time distribution in periods')

    plt.subplot(223)
    x = []
    y = []
    for k, v in model_results['pmf_wait_time'].items():
        if k >= 1:
            x.append(k)
            y.append(v)
    plt.bar(x, y)
    plt.xticks([1, 2, 3], [1, 2, 3])
    plt.xlabel('Time periods')
    plt.ylabel('Probability')
    plt.title('Waiting time distribution')

    plt.subplot(224)
    x = []
    y = []
    for k, v in model_results['pmf_shortfall'].items():
        x.append(k)
        y.append(v)
    plt.bar(x, y)
    plt.xticks([1, 5, 10, 15, 20, 25], [1, 5, 10, 15, 20, 25])
    plt.xlabel('Units')
    plt.ylabel('Probability')
    plt.title('Shortfall distribution')
    plt.show()
