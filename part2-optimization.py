"""## Part 2 - Optimization

### How to speed up optimization
Gradient descent is an iterative optimization method. The log-linear objective presented in equation (1) is a convex problem, which guaranties a convergence for gradient descent iterations (when choosing an appropriate step size). However, in order to converge, many iterations must be performed. Therefore, it is importent to (1) speed up each iteration and to (2) decrease the number of iterations.


#### Decrease the number of iterations
Notice that by increasing $\lambda$ we can force the algorithm to search for a solution in a smaller search space - which will reduce the number of iterations. However, this is a tredoff, because it will also damage train-set accuracy (Notice that we don't strive to achieve 100% accuracy on the training set, as sometimes by reducing training accuracy we achieve improvement in developement set accuracy).


#### Decrease iteration duration
Denote the GD update as:

In this excersice we are using `fmin_l_bfgs_b`, which is imported from `scipy.optimize`. This is an iterative optimization function which is similar to GD. The function receives 3 arguments:


1.   **func** - a function that clculates the objective and its gradient each iteration.
2.   **x0** - initialize values of the model weights.
3.   **args** - the arguments which 'func' is expecting, except for the first argument - which is the model weights.
4.   **maxiter** (optional) - specify a hard limit for number of iterations. (int)
5.   **iprint**  (optional) - specify the print interval (int) 0 < iprint < 99



Think of ways to efficiently calculate eqautions (1) and (2) according to your features implementation. Furthermore, think which parts must be computed in each iteration, and whether others can be computed once.
"""


def calc_objective_per_iter(w_i, arg_1, arg_2, ...):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param w_i: weights vector in iteration i
        :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """

    ## Calculate the terms required for the likelihood and gradient calculations
    ## Try implementing it as efficient as possible, as this is repeated for each iteration of optimization.

    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    return (-1) * likelihood, (-1) * grad


"""Now lets run the code untill we get the optimized weights."""

from scipy.optimize import fmin_l_bfgs_b

# Statistics
statistics = feature_statistics_class()
statistics.get_word_tag_pairs(train_path)

# feature2id
feature2id = feature2id_class(statistics, threshold)
feature2id.get_word_tag_pairs(train_path)

# define 'args', that holds the arguments arg_1, arg_2, ... for 'calc_objective_per_iter'
args = (arg_1, arg_2, ...)
w_0 = np.zeros(n_total_features, dtype=np.float32)
optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50)
weights = optimal_params[0]

# Now you can save weights using pickle.dump() - 'weights_path' specifies where the weight file will be saved.
# IMPORTANT - we expect to recieve weights in 'pickle' format, don't use any other format!!
weights_path = 'your_path_to_weights_dir/trained_weights_data_i.pkl'  # i identifies which dataset this is trained on
with open(weights_path, 'wb') as f:
    pickle.dump(optimal_params, f)

#### In order to load pre-trained weights, just use the next code: ####
#                                                                     #
# with open(weights_path, 'rb') as f:                                 #
#   optimal_params = pickle.load(f)                                   #
# pre_trained_weights = optimal_params[0]                             #
#                                                                     #
#######################################################################

