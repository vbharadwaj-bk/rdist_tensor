from distmat import *
import numpy as np
import numpy.linalg as la
import json

import cppimport.import_hook
import cpp_ext.tensor_kernels as tensor_kernels 

# Initializes a distributed tensor of a known low rank
class DistLowRank:
    def __init__(self, tensor_grid, rank, singular_values): 
        self.rank = rank
        self.grid = tensor_grid.grid
        self.tensor_grid = tensor_grid
        self.mode_sizes = tensor_grid.tensor_dims

        self.dim = len(tensor_grid.tensor_dims)
        self.factors = [DistMat1D(rank, tensor_grid, i)
                    for i in range(self.dim)]

        # TODO: This array is currently useless, need to fix! 
        self.singular_values = np.array(singular_values)
        
    # Replicate data on each slice and all-gather all factors accoording to the
    # provided true / false array 
    def allgather_factors(self, which_factors):
        gathered_matrices = []
        for i in range(self.dim):
            if which_factors[i]:
                self.factors[i].allgather_factor()
                gathered_matrices.append(self.factors[i].gathered_factor)

        return gathered_matrices, None

    def materialize_tensor(self):
        '''
        TODO: THIS FUNCTION IS CURRENTLY BROKEN!
        '''
        gathered_matrices, _ = self.allgather_factors([True] * self.dim)
        self.local_materialized = tensor_from_factors_sval(self.singular_values, gathered_matrices) 

    # Materialize the tensor starting only from the given singular
    # value 
    def partial_materialize(self, singular_value_start):
        assert(singular_value_start < self.rank)

        gathered_matrices, _ = self.allgather_factors([True] * self.dim)
        gathered_matrices = [mat[:, singular_value_start:] for mat in gathered_matrices]
        
        return tensor_from_factors_sval(self.singular_values[singular_value_start:], gathered_matrices) 

    def initialize_factors_deterministic(self, offset):
        for factor in self.factors:
            factor.initialize_deterministic(offset)

    def initialize_factors_random(self, random_seed=42):
        for factor in self.factors:
            factor.initialize_random(random_seed=random_seed)

    def compute_tensor_values(self, idxs):
        '''
        Calls into the C++ layer to compute the value of the low rank tensor
        at specific indices
        '''
        result = np.zeros(len(idxs[0]), dtype=np.double)
        gathered_matrices, _ = self.allgather_factors([True] * self.dim)
        tensor_kernels.compute_tensor_values(gathered_matrices, idxs, result)
        return result

    def compute_loss(self, ground_truth, alpha=0.5):
        '''
        Ground truth: an instance of the tensor to decompose
        nonzero_sample_frac: Fraction of nonzeros to sample;
                             This param is currently not implemented!
                             We instead compute the loss on all of
                             the nonzeros
        alpha: proportion of zeros to nonzeros in the sample.

        TODO: Alpha is funged a bit in this computation, since we throw
        out any zero samples that turn out to be nonzeros and recompute the
        denominator. This computation is still valid with a slightly
        different value of alpha, but we should eventually fix this.
        '''
        if ground_truth.type == "SPARSE_TENSOR": 
            tg = self.tensor_grid
            nnz_to_sample = ground_truth.nnz

            # Compute the loss on the nonzeros 
            lr_values = self.compute_tensor_values(ground_truth.tensor_idxs) 
            nonzero_loss = get_norm_distributed(ground_truth.values - lr_values, self.grid.comm)

            # Compute the loss on the zeros
            box_dims = [tg.bound_ends[j] - tg.bound_starts[j] for j in range(self.dim)]
            logsum_box_dims = np.sum(np.log(box_dims))
            logsum_full_ten_dims = np.sum(np.log(tg.tensor_dims)) 

            vol_frac_owned = np.exp(logsum_box_dims - logsum_full_ten_dims)

            # TODO: To get exactly the same distribution, we should
            # really compute this value in two stages using a multinomial
            # distribution... 
            zeros_to_sample_global = ((1 / alpha) - 1) * ground_truth.nnz
            local_zeros_to_sample = (vol_frac_owned * zeros_to_sample_global).astype(np.ulonglong)
            remaining = local_zeros_to_sample
            zero_loss = 0.0

            # TODO: Should put this in a while loop! 
            zero_samples = [] 
            for j in range(self.dim):
                start = tg.bound_starts[j]
                end = tg.bound_ends[j]

                idxs = np.random.randint(0, end - start, size=remaining, dtype=np.ulonglong) 
                zero_samples.append(idxs)

            collisions = ground_truth.idx_filter.check_idxs(zero_samples)
            zero_values = self.compute_tensor_values(zero_samples)


            zero_values[collisions] = 0.0
            zero_loss += la.norm(zero_values) ** 2

            # Exact computation of alpha and sfit here

            local_zeros_to_sample -= len(collisions)
            true_zero_count = self.grid.comm.allreduce(local_zeros_to_sample)

            sfit = ground_truth.nnz + true_zero_count
            alpha = ground_truth.nnz / sfit 

            nonzero_loss = (nonzero_loss ** 2) * ground_truth.nnz / np.ceil(alpha * sfit)
            zero_loss = self.grid.comm.allreduce(zero_loss)

            dense_entries = np.prod(np.array(self.tensor_grid.tensor_dims, dtype=np.double))
            zero_loss *= (dense_entries - ground_truth.nnz) / np.floor((1 - alpha) * sfit)
            estimated_fit = 1 - (np.sqrt(nonzero_loss + zero_loss) / ground_truth.tensor_norm) 
            return estimated_fit
        else:
            assert False 

    # Computes a distributed MTTKRP of all but one of this 
    # class's factors with a given dense tensor. Also performs 
    # gram matrix computation. 
    def optimize_factor(self, local_ten, mode_to_leave, timer_dict, sketching_pct=None):
        sketching = sketching_pct is not None
        factors_to_gather = [True] * self.dim
        factors_to_gather[mode_to_leave] = False

        selected_indices = np.array(list(range(self.dim)))[factors_to_gather]
        selected_factors = [self.factors[idx] for idx in selected_indices] 

        # Compute gram matrices of all factors but the one we are currently
        # optimizing for, perform leverage-score based sketching if necessary 
        for idx in selected_indices:
            factor = self.factors[idx]
            
            start = start_clock()
            factor.compute_gram_matrix()
            stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

            if sketching_pct is not None:
                start = start_clock() 
                factor.compute_leverage_scores()
                stop_clock_and_add(start, timer_dict, "Leverage Score Computation")


        start = start_clock() 
        gram_prod = selected_factors[0].gram

        for i in range(1, len(selected_factors)):
            gram_prod = np.multiply(gram_prod, selected_factors[i].gram)

        # Compute inverse of the gram matrix 
        krp_gram_inv = la.pinv(gram_prod)
        stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

        start = start_clock() 
        gathered_matrices, gathered_leverage = self.allgather_factors(factors_to_gather)

        dummy = None
        gathered_matrices.insert(mode_to_leave, dummy)

        stop_clock_and_add(start, timer_dict, "Slice Replication")

        MPI.COMM_WORLD.Barrier()
        start = start_clock() 
        mttkrp_unreduced = np.zeros((self.tensor_grid.intervals[mode_to_leave], self.rank)) 
        local_ten.mttkrp(gathered_matrices, mode_to_leave, mttkrp_unreduced)
        MPI.COMM_WORLD.Barrier()
        stop_clock_and_add(start, timer_dict, "MTTKRP")
        start = start_clock()

        mttkrp_reduced = np.zeros_like(self.factors[mode_to_leave].data)

        self.grid.slices[mode_to_leave].Reduce_scatter([mttkrp_unreduced, MPI.DOUBLE], 
                [mttkrp_reduced, MPI.DOUBLE])  
        stop_clock_and_add(start, timer_dict, "Slice Reduce-Scatter")

        start = start_clock() 
        res = (krp_gram_inv @ mttkrp_reduced.T).T.copy()
        self.factors[mode_to_leave].data = res
        stop_clock_and_add(start, timer_dict, "Gram-Times-MTTKRP")

        return timer_dict

    def als_fit(self, local_ground_truth, num_iterations, sketching_pct, output_file, compute_accuracy=False):
        # Should initialize the singular values more intelligently, but this is fine
        # for now:

        statistics = {
                        "Gram Matrix Computation": 0.0,
                        "Leverage Score Computation": 0.0,
                        "Slice Replication": 0.0,
                        "MTTKRP": 0.0,
                        "Slice Reduce-Scatter": 0.0,
                        "Gram-Times-MTTKRP": 0.0
                        }

        statistics["Mode Sizes"] = self.mode_sizes.tolist()
        statistics["Tensor Target Rank"] = self.rank
        statistics["Processor Count"] = self.grid.world_size
        statistics["Grid Dimensions"] = self.grid.axesLengths.tolist()

        self.singular_values = np.ones(self.rank)

        for iter in range(num_iterations):
            if compute_accuracy:
                loss = self.compute_loss(local_ground_truth) 

                if self.grid.rank == 0:
                    print("Estimated Fit after iteration {}: {}".format(iter, loss)) 

            for mode_to_optimize in range(self.dim):
                self.optimize_factor(local_ground_truth, mode_to_optimize, statistics, sketching_pct=sketching_pct)

        if self.grid.rank == 0:
            f = open(output_file, 'a')
            print(statistics)
            json_obj = json.dumps(statistics, indent=4)
            f.write(json_obj + ",\n")
            f.close()

if __name__=='__main__':
    pass