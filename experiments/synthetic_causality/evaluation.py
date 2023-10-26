import numpy as np
from tqdm import tqdm


def evaluate_intervention(sem, model, train_data, n_eval_points, n_dist_samp):
    """Evaluate interventional distribution for all variables.

    Args:
        ...
        n_eval_points (int): Number of interventional values to compute.
        n_dist_samp (int): Number of flow samples used to compute distribution.
    """
    emp_means = np.mean(train_data, axis=0)

    # Stores interventional values
    true_mat = np.empty((sem.n_var, sem.n_var, n_eval_points))
    true_mat[:] = np.nan

    pred_mat = np.empty((sem.n_var, sem.n_var, n_eval_points))
    pred_mat[:] = np.nan

    # Stores evaluation points for plotting
    val_mat = np.empty((sem.n_var, n_eval_points))

    for int_ind in range(sem.n_var):
        var_mean = emp_means[int_ind]

        pad = n_eval_points // 2
        eval_vals = np.arange(int(var_mean) - pad, int(var_mean) + pad, step=1)

        val_mat[int_ind] = eval_vals

        for i, int_val in enumerate(eval_vals):
            int_input = [None] * sem.n_var
            int_input[int_ind] = int_val

            true_dist = sem.generate_int_dist(int_input, n_dist_samp)

            while True:
                try:
                    p_dist = model.predict_intervention_modified(int_val,
                                                                 n_dist_samp,
                                                                 iidx=int_ind)
                    p_dist = p_dist[0]
                except ValueError:
                    continue
                except AssertionError:
                    continue
                break

            # Exclude interventional variable, and preceding variables
            true_mat[int_ind+1:, int_ind, i] = true_dist[int_ind+1:]
            pred_mat[int_ind+1:, int_ind, i] = p_dist[int_ind+1:]

    return true_mat, pred_mat, val_mat


def evaluate_counterfactual(sem, model, train_data, n_eval_points, n_cf_samp):
    emp_means = np.mean(train_data, axis=0)

    # Stores counterfactual values
    err_mat = np.empty((sem.n_var, sem.n_var, n_eval_points))
    err_mat[:] = np.nan

    for cf_ind in tqdm(range(sem.n_var)):
        var_mean = emp_means[cf_ind]

        pad = n_eval_points // 2
        eval_vals = np.arange(int(var_mean) - pad, int(var_mean) + pad, step=1)

        for i, cf_val in enumerate(eval_vals):
            cf_input = [None] * sem.n_var
            cf_input[cf_ind] = cf_val

            errors = []
            for _ in range(n_cf_samp):
                obs, e = sem.generate_ctf_obs()
                
                ctf_true = sem.generate_counterfactual(e, cf_input)
                
                p_dist = model.predict_counterfactual(obs.reshape(1,-1),
                                                      cf_val,
                                                      iidx=cf_ind)[0]
                # Only evaluate error past cf index
                error = sum((ctf_true[cf_ind:] - p_dist[cf_ind:]) ** 2)
                errors.append(error)
            
            err_mat[cf_ind+1:, cf_ind, i] = np.mean(errors)

    return err_mat


def get_nrmse(true_dist, pred_dist):
    """Get the normalized root MSE in estimation per variable."""
    sq_error = ((true_dist - pred_dist) ** 2)

    # Average error across evaluations
    var_err = np.mean(sq_error, axis=-1)

    # Normalize
    var_err = np.sqrt(var_err)
    var_err /= (np.max(true_dist, axis=-1) - np.min(true_dist, axis=-1))

    # Average error per intervened variable
    avg_err = np.nanmean(var_err, axis=0)

    return avg_err, var_err


def get_mse(true_dist, pred_dist):
    sq_error = ((true_dist - pred_dist) ** 2)

    # Average error across evaluations
    var_err = np.mean(sq_error, axis=-1)

    # Average error per intervened variable
    avg_err = np.nanmean(var_err, axis=0)

    return avg_err, var_err