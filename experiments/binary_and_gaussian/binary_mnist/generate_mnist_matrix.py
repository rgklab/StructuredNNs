import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from generate_binary_data import DataGenerator

LABEL_SIZE = 20
TITLE_SIZE = 32
SUB_TITLE_SIZE = 20
MARKER_SIZE = 20


def obtain_adjacency_neighbors(shp, nbr_size, dense_centre=None, print_nbr_sizes=False, debug_list=None):
    """
    For an image of shp*shp, and a prespecified adjacency distance, return the adjacency matrix
    corresponding to a grid structured conditional independence map

    @param shp: length/width of the image
    @param nbr_size: radius of the connection square around each pixel
    @param dense_centre: if None, just use fixed nbr_size for all pixels
        if specified, dense_centre must be of the form {border_size: nbr_size, ...}
    @param debug_list: list of nodes (i,j) for which to print the
        corresponding connectivity matrix
    @return: shp**2 by shp**2 adjacency matrix
    """
    if dense_centre:
        assert isinstance(dense_centre, dict), 'Invalid dense_centre type!'
        border_sizes = list(dense_centre.keys())
        border_sizes.sort(reverse=True)

    nbr_set = {}
    for i in range(shp):
        for j in range(shp):
            nbr_set[(i, j)] = []

            if dense_centre:
                p_nbr_size = nbr_size
                # Set correct nbr_size
                for K in border_sizes:
                    assert K <= shp, "Border size greater than img dimensions!"
                    if i >= K and i <= shp - K - 1 and j >= K and j <= shp - K - 1:
                        p_nbr_size = dense_centre[K]
                    continue
            else:
                p_nbr_size = nbr_size

            if print_nbr_sizes:
                print(i, j, p_nbr_size)

            # Find valid ancestors
            for a in range(i - p_nbr_size, i + p_nbr_size + 1):
                for b in range(j - p_nbr_size, j + p_nbr_size + 1):
                    if a >= 0 and a < shp and b >= 0 and b < shp and (a, b) != (i, j):
                        nbr_set[(i, j)].append((a, b))

    if debug_list:
        for tv in debug_list:
            D = np.zeros((shp, shp))
            for t in nbr_set[tv]:
                D[t[0], t[1]] = 1
            print(D)

    # Create adjacency matrix using parent set
    A = np.zeros((shp * shp, shp * shp))
    for i in range(shp):
        for j in range(shp):
            idx_set = nbr_set[(i, j)]
            row_vals = [k[0] for k in idx_set]
            col_vals = [k[1] for k in idx_set]
            parents = np.ravel_multi_index(np.array([row_vals, col_vals]), (shp, shp))
            node = np.ravel_multi_index(np.array([[i], [j]]), (shp, shp))
            for p in parents:
                if p < node:
                    A[p, node] = 1

    if debug_list:
        row_vals = [k[0] for k in debug_list]
        col_vals = [k[1] for k in debug_list]
        node_idx = np.ravel_multi_index(np.array([row_vals, col_vals]), (shp, shp))
        print(node_idx)

    return A.T


"""
Functions to optimize masks
"""


def optimize_all_masks(opt_type, hidden_sizes, A, var_penalty_weight=0.0):
    # Function returns mask list in order for layers from inputs to outputs
    # This order matches how the masks are assigned to the networks in MADE
    masks = []
    constraint = np.copy(A)
    for l in hidden_sizes:
        if opt_type == 'greedy':
            (M1, M2) = optimize_single_mask_greedy(constraint, l)
        elif opt_type == 'IP':
            (M1, M2) = optimize_single_mask_gurobi(constraint, l)
        elif opt_type == 'IP_alt':
            (M1, M2) = optimize_single_mask_gurobi(constraint, l, alt=True)
        elif opt_type == 'LP_relax':
            (M1, M2) = optimize_single_mask_gurobi(constraint, l, relax=True)
        elif opt_type == 'IP_var':
            (M1, M2) = optimize_single_mask_gurobi(
                constraint, l, var_pen=True, var_penalty_weight=var_penalty_weight
            )
        else:
            raise ValueError('opt_type is not recognized: ' + str(opt_type))

        constraint = M1
        masks = masks + [M2.T]  # take transpose for size: (n_inputs x n_hidden/n_output)
    masks = masks + [M1.T]
    return masks


def optimize_single_mask_greedy(A, n_hidden):
    # decompose A as M1 * M2
    # A size: (n_outputs x n_inputs)
    # M1 size: (n_outputs x n_hidden)
    # M2 size: (n_hidden x n_inputs)

    # find non-zero rows and define M2
    A_nonzero = A[~np.all(A == 0, axis=1), :]
    n_nonzero_rows = A_nonzero.shape[0]
    M2 = np.zeros((n_hidden, A.shape[1]))
    for i in range(n_hidden):
        M2[i, :] = A_nonzero[i % n_nonzero_rows]

    # find each row of M1
    M1 = np.ones((A.shape[0], n_hidden))
    for i in range(M1.shape[0]):
        # Find indices where A is zero on the ith row
        Ai_zero = np.where(A[i, :] == 0)[0]
        # find row using linear programming + rounding
        # res = linprog(-1*np.ones(n_hidden), A_eq=M2[:,Ai_zero].T, b_eq=np.zeros(len(Ai_zero)),
        #              bounds=np.vstack((np.zeros(n_hidden), np.ones(n_hidden))).T)#, method='revised simplex')
        # M1[i,:] = np.round(res.x)
        # find row using closed-form solution
        # find unique entries (rows) in j-th columns of M2 where Aij = 0
        row_idx = np.unique(np.where(M2[:, Ai_zero] == 1)[0])
        M1[i, row_idx] = 0.0

    return M1, M2


def optimize_single_mask_mouton(A, n_hidden):
    """
    Algo proposed in the Mouton workshop paper
    Randomly samples rows from A for the mask
    @param A:
    @param n_hidden:
    @return:
    """
    pass


def optimize_single_mask_ian(A, n_hidden):
    """
    Ian's "optimal" factorization algo
    Picks rows from A that introduce the most connections for the mask
    @param A:
    @param n_hidden:
    @return:
    """
    pass


def variance(data):
    n = len(data)
    mean = sum(data) / n
    squared_diff_sum = gp.LinExpr()
    for x in data:
        squared_diff_sum += gp.QuadExpr((x - mean) * (x - mean))
    variance = squared_diff_sum / n
    return variance


def diff(data):
    return max(data) - min(data)


def optimize_single_mask_gurobi(A, n_hidden, alt=False, relax=False, var_pen=False, var_penalty_weight=0.0):
    try:
        with gp.Env(empty=True) as env:
            env.setParam('LogToConsole', 0)
            env.start()
            with gp.Model(env=env) as m:
                # Create variables
                if relax:
                    # LP relaxation
                    M1 = m.addMVar((A.shape[0], n_hidden), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="M1")
                    M2 = m.addMVar((n_hidden, A.shape[1]), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="M2")
                    m.params.NonConvex = 2
                else:
                    # Original integer program
                    M1 = m.addMVar((A.shape[0], n_hidden), lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="M1")
                    M2 = m.addMVar((n_hidden, A.shape[1]), lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="M2")

                # Set constraints and objective
                if alt:
                    # Alternate formulation: might violate adjacency structure
                    m.setObjective(
                        sum(M1[i, :] @ M2[:, j] for i in range(A.shape[0]) for j in range(A.shape[1]) if A[i, j] == 1) - \
                        sum(M1[i, :] @ M2[:, j] for i in range(A.shape[0]) for j in range(A.shape[1]) if A[i, j] == 0),
                        GRB.MAXIMIZE
                    )
                else:
                    # Original formulation: guarantees adjacency structure is respected
                    m.addConstrs(
                        (M1[i, :] @ M2[:, j] <= A[i, j] for i in range(A.shape[0]) for j in range(A.shape[1]) if
                         A[i, j] == 0),
                        name='matrixconstraints')
                    m.addConstrs(
                        (M1[i, :] @ M2[:, j] >= A[i, j] for i in range(A.shape[0]) for j in range(A.shape[1]) if
                         A[i, j] > 0),
                        name='matrixconstraints2')

                    if var_pen:
                        # Variance-penalized objective
                        # m.setObjective(sum(A_prime) - diff(A_prime), GRB.MAXIMIZE) #<- doesn't work - can't multiply QuadExpr
                        A_prime = {}
                        for i in range(A.shape[0]):
                            A_prime[i] = {}
                            for j in range(A.shape[1]):
                                A_prime[i][j] = m.addVar(lb=0.0, ub=A.shape[0], obj=0.0, vtype=GRB.INTEGER,
                                                         name=f"A_prime_{i}{j}")
                                m.addConstr((A_prime[i][j] == M1[i, :] @ M2[:, j]), name=f'constr_A_prime_{i}{j}')
                        m.setObjective(sum(A_prime) - var_penalty_weight * variance(A_prime), GRB.MAXIMIZE)
                    else:
                        # Original objective
                        A_prime = [M1[i, :] @ M2[:, j] for i in range(A.shape[0]) for j in range(A.shape[1])]
                        m.setObjective(sum(A_prime), GRB.MAXIMIZE)

                # Optimize model
                m.optimize()
                obj_val = m.getObjective().getValue()

                # Obtain optimized results
                result = {}
                result['M1'] = np.zeros((A.shape[0], n_hidden))
                result['M2'] = np.zeros((n_hidden, A.shape[1]))
                for v in m.getVars():
                    if v.varName[0] == 'M':
                        nm = v.varName.split('[')[0]
                        idx = (v.varName.split('[')[1].replace(']', '')).split(',')
                        col_idx = int(idx[0])
                        row_idx = int(idx[1])
                        if relax:
                            # Round real-valued solution to nearest integer
                            val = v.x
                            if val <= 0.5:
                                result[nm][col_idx][row_idx] = 0
                            else:
                                result[nm][col_idx][row_idx] = 1
                        else:
                            result[nm][col_idx][row_idx] = v.x

        print(f'Successful opt! obj = {obj_val}')
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        print('Encountered an attribute error')
    return result['M1'], result['M2']


matrix_hash = lambda A: tuple([tuple(k.ravel().tolist()) for k in np.split(A, A.shape[-1])])


def generate_adj_mtx(d, threshold):
    attempt_num = 0
    while True:
        # Parse adj_type for how many entries to randomly zero out

        # Initialize randomly
        print(f"Adj mtx generation attempt {attempt_num}:")
        A = np.random.standard_normal((d, d))
        # Threshold and make lower triangular
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j] > -1 * threshold and A[i][j] < threshold:
                    A[i][j] = 0
                else:
                    A[i][j] = 1
        A = np.tril(A, -1)

        # Check each node has dependencies
        sums = A.sum(axis=1)
        has_empty_row = False
        for sum in sums[1:]:
            if sum == 0:
                has_empty_row = True
        if has_empty_row:
            print("Attempt failed; retry ...")
            attempt_num += 1
            continue

        print("Attempt successful - random adj mtx generated!")
        return A


def mask_prod(mask_list, reverse=False):
    """
    Note: Our MADE setup requires reverse=False
        Use reverse=True for Zuko
    @param mask_list: [M1, M2,...,Ml]
    @param reverse:
    @return:
    """
    if reverse:
        # Ml @ ... @ M2 @ M1
        prod = masks[-1]
        for i in range(len(masks) - 2, -1, -1):
            prod = prod @ masks[i]
        return prod
    else:
        # (M1 @ M2 @ ... @ Ml).T
        prod = mask_list[0]
        for i in range(1, len(mask_list)):
            prod = prod @ mask_list[i]
        return prod.T


def check_masks(mask_list, A):
    """
    Given masks [M1.T, M2.T, ..., Mk.T], check if the matrix product
    Mk*...*M2*M1 respects A's adjacency structure

    @param mask_list: list of masks
    @param A: adjacency matrix
    @return: True if list of masks preserve A's adjacency constraints
    """
    mask_prod = (mask_list[0] @ mask_list[1]).T
    constraint = (mask_prod > 0.0001) * 1. - A
    if np.any(constraint != 0.):
        return False
    else:
        return True


def LP_relax_test(num_tests):
    success = 0

    for i in range(num_tests):
        print(f'Test {i} out of {num_tests} ... ')
        h_multiplier = np.random.randint(low=1, high=5)
        dim = 10
        threshold = np.random.rand()
        A = generate_adj_mtx(dim, threshold)
        hidden_sizes = (h_multiplier * dim,)
        masks = optimize_all_masks('LP_relax', hidden_sizes, A)

        if check_masks(masks, A):
            success += 1

    print(f'{success} successes out of {num_tests} trials, {success / num_tests}')


def poster_mask_experiment(load_results=False):
    dim = 15
    h_multiplier = 3
    ip_num_trials = 2
    greedy_num_trials = 100
    thresholds = [x / 10 for x in range(1, 10)]
    opt_types = ['greedy', 'IP']

    if load_results:
        with open(f'poster_masks_results_d{dim}.txt', 'rb') as file:
            results = pickle.load(file)
    else:
        results = {}
        results['time'] = {}
        results['connections'] = {}

        # Do timing experiments first
        for threshold in thresholds:
            results['time'][threshold] = {}
            results['time'][threshold]['IP'] = 0
            results['time'][threshold]['greedy'] = 0

            for i in range(ip_num_trials):
                # IP
                A = generate_adj_mtx(dim, threshold)
                hidden_sizes = (h_multiplier * dim,)
                print(f'IP - hiddens: {hidden_sizes[0]}, threshold: {threshold}, trial #{i}')
                start_time = datetime.now()
                masks = optimize_all_masks('IP', hidden_sizes, A)
                end_time = datetime.now()

                results['time'][threshold]['IP'] \
                    += (end_time - start_time).total_seconds()
            results['time'][threshold]['IP'] /= ip_num_trials

            for i in range(greedy_num_trials):
                # Greedy
                ## REMOVE LATER: need to increase dims or it underflows ##
                A = generate_adj_mtx(dim * 3, threshold)
                hidden_sizes = (h_multiplier * dim * 3,)
                print(f'Greedy - hiddens: {hidden_sizes[0]}, threshold: {threshold}, trial #{i}')
                start_time = datetime.now()
                masks = optimize_all_masks('greedy', hidden_sizes, A)
                end_time = datetime.now()
                results['time'][threshold]['greedy'] \
                    += (end_time - start_time).total_seconds()

        # Max connections experiments
        for threshold in thresholds:
            results['connections'][threshold] = {}
            results['connections'][threshold]['IP'] = 0
            results['connections'][threshold]['greedy'] = 0

            A = generate_adj_mtx(dim, threshold)
            hidden_sizes = (h_multiplier * dim,)

            for opt_type in opt_types:
                masks = optimize_all_masks(opt_type, hidden_sizes, A)
                results['connections'][threshold][opt_type] \
                    += sum(sum(masks[1].T @ masks[0].T))

        # Save results for future use
        with open(f'poster_masks_results_d{dim}.txt', 'wb') as file:
            pickle.dump(results, file)

    # Plot results
    # time_results = results['time']
    # conn_results = results['connections']
    fig, axs = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={'wspace': 0.1}
    )
    for idx, category in enumerate(['time', 'connections']):
        data = results[category]
        df = pd.DataFrame(data).transpose()
        if category == 'time':
            # Report execution time in base log10
            df = np.log10(df)
        ax = axs[idx]
        df.plot(kind='bar', ax=ax)
        if idx == 1:
            ax.set_xlabel("Sparsity Threshold", fontsize=LABEL_SIZE)
        if category == 'connections':
            ax.set_ylabel("# Connections", fontsize=LABEL_SIZE)
        else:
            ax.set_ylabel('Execution Time (log10)', fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        # ax.set_title(f"# Hiddens = {h_multiplier * dim}", fontsize=SUB_TITLE_SIZE)
        ax.legend(title='Opt Method', fontsize=LABEL_SIZE)

    plt.savefig(f'poster_mask_comparison_d{dim}.pdf')


def mask_experiment(load_results=False):
    h_multipliers = range(1, 5)
    dim = 10
    thresholds = [x / 10 for x in range(1, 10)]
    num_trials = 2
    # opt_types = ['greedy', 'IP', 'IP_alt', 'LP_relax']
    opt_types = ['greedy', 'IP']

    if load_results:
        with open(f'mask_comparison_results_d{dim}.txt', 'rb') as file:
            results = pickle.load(file)
    else:
        results = {}

        for h_multiplier in h_multipliers:
            results[h_multiplier] = {}
            hidden_sizes = (h_multiplier * dim,)

            for threshold in thresholds:
                results[h_multiplier][threshold] = {}

                for opt_type in opt_types:
                    results[h_multiplier][threshold][opt_type] = 0

                for i in range(num_trials):
                    A = generate_adj_mtx(dim, threshold)

                    for opt_type in opt_types:
                        print(f'hiddens: {hidden_sizes[0]}, threshold: {threshold}, opt_type: {opt_type}, trial #{i}')
                        masks = optimize_all_masks(opt_type, hidden_sizes, A)
                        # check_masks(masks, A, opt_type)

                        results[h_multiplier][threshold][opt_type] += sum(sum(masks[1].T @ masks[0].T))

                for opt_type in opt_types:
                    results[h_multiplier][threshold][opt_type] /= num_trials

        # Save results for future use
        with open(f'mask_comparison_results_d{dim}.txt', 'wb') as file:
            pickle.dump(results, file)

    # Plot results
    fig, axs = plt.subplots(
        1, 4, figsize=(45, 8), gridspec_kw={'wspace': 0.2}
    )
    for idx, h_multiplier in enumerate(h_multipliers):
        data = results[h_multiplier]
        df = pd.DataFrame(data).transpose()
        ax = axs[idx]
        df.plot(kind='bar', ax=ax)
        ax.set_xlabel("Sparsity Threshold", fontsize=LABEL_SIZE)
        if idx == 0:
            ax.set_ylabel("# Connections", fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        ax.set_title(f"# Hiddens = {h_multiplier * dim}", fontsize=SUB_TITLE_SIZE)
        ax.legend(title='Opt Method', fontsize=LABEL_SIZE)

    fig.suptitle(
        'Comparison of Greedy and Exact Factorization Methods Across Different Sparsity Levels',
        fontsize=TITLE_SIZE
    )
    # plt.show()
    plt.savefig(f'mask_comparison_d{dim}.png')


def straf_experiment():
    A = np.ones((4, 4))
    A = A = np.tril(A, -1)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j != 0:
                A[i][j] = 0
    hidden_sizes = (5,)

    for opt_type in ['greedy', 'IP', 'IP_var']:
        masks = optimize_all_masks(opt_type, hidden_sizes, A)
        A1 = masks[1].T
        A0 = masks[0].T
        print(opt_type)
        import pdb;
        pdb.set_trace()


if __name__ == '__main__':
    try:
        # Generate adjacency matrices for MNIST
        for nbr_size in [10, 12, 15, 18, 20]:
            A = obtain_adjacency_neighbors(shp=28, nbr_size=nbr_size)
            np.savez(f'./synth_data_files/mnist/28_nbr{nbr_size}_adj', A)

        ## Zuko mask comparisons
        adjacency = np.tril(np.ones(10), -1)
        adjacency[4][1] = 0
        adjacency[3][1] = 0
        adjacency[4][2] = 0
        print(adjacency)
        masks = optimize_all_masks('greedy', [10, 10, 10], adjacency)
        prod = masks[0]
        for i in range(1, len(masks)):
            prod = prod @ masks[i]
        print(prod.T)
        print(sum(sum(prod)))
    except:
        import sys, pdb, traceback

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)



