import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler
from scipy.optimize import minimize
from botorch.models.transforms.outcome import Standardize


class BayesianOptimizer:
    def __init__(self, dimension=2):
        self.device = self.call_device()
        self.dtype = torch.float64
        self.bounds = torch.tensor([[0.0] * dimension, [1.0] * dimension], device=self.device, dtype=self.dtype)
        self.observed_x = []
        self.D = []
        self.goodness = []
        self.BATCH_SIZE = 3
        self.NUM_RESTARTS = 10
        self.RAW_SAMPLES = 512
        self.model = None
        self.mll = None
        self.qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

    def call_device(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        device = torch.device("cpu")
        return device
    
    def observe_behaivor_estimate(self, chosen_option, other_options):
        def get_index(option):
            if option not in self.observed_x:
                self.observed_x.append(option)
            return self.observed_x.index(option)
        chosen_index = get_index(chosen_option)
        other_indices = [get_index(option) for option in other_options]
        new_entry = [chosen_index] + other_indices
        self.D.append(new_entry)
        self.goodness = self.perform_map_estimation(self.D)
        print("estimated:",self.goodness.tolist())

    def calc_btl_likelihood(self, preference, y, scale=1.0):
            tmp = y[preference]
            log_prob = np.log(np.exp(tmp[0] / scale) / np.sum(np.exp(tmp / scale)))
            return log_prob

    def objective(self, y, preferences, scale=1.0):
        total_log_likelihood = 0.0
        for preference in preferences:
            total_log_likelihood += self.calc_btl_likelihood(preference, y, scale)
        return -total_log_likelihood

    def perform_map_estimation(self, preferences, initial_y=None, scale=0.01, num_iters=100):
        if initial_y is None:
            # initial_y = np.ones(len(self.X))
            n = len(self.observed_x)
            initial_y = np.ones(n) / n  # uniform distribution

        result = minimize(self.objective, initial_y, args=(preferences, scale), 
                            method='L-BFGS-B', bounds=[(0, None)] * n)
        optimal_y = result.x
        prob_y = np.exp(optimal_y) / np.sum(np.exp(optimal_y))
        
        return prob_y
    
    def initialize_model(self, train_x, train_y):
        # GP 모델 초기화
        # if isinstance(train_x, np.ndarray):
        #     train_x = train_x.astype(np.float32)
        # if isinstance(train_y, np.ndarray):
        #     train_y = train_y.astype(np.float32)

        
        norm_x = torch.tensor(train_x, dtype=self.dtype, device=self.device)
        norm_y = torch.tensor(train_y, dtype=self.dtype, device=self.device)

        norm_y = norm_y.unsqueeze(1)
        self.model = SingleTaskGP(norm_x, norm_y, covar_module=ScaleKernel(MaternKernel(nu=2.5)), outcome_transform=Standardize(m=1))
        self.model = self.model.to(dtype=torch.float64, device=self.device)
        # MLL 초기화
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.mll = self.mll.to(dtype=torch.float64, device=self.device)
        # 모델 학습
        fit_gpytorch_mll(self.mll)

    def optimize_acqf_and_get_observation(self):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        self.initialize_model(self.observed_x, self.goodness)
        norm_x = torch.tensor(self.observed_x, dtype=self.dtype, device=self.device)

        qLogNEI = qLogNoisyExpectedImprovement(
            model=self.model,
            X_baseline=norm_x,
            sampler=self.qmc_sampler,
        )

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=qLogNEI,
            bounds=self.bounds,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        # observe new values
        new_x = candidates.detach()
        return new_x
    

# optimizer_1 = BayesianOptimizer(dimension=3)
# optimizer_2 = BayesianOptimizer(dimension=3)
# optimizer_3 = BayesianOptimizer(dimension=3)
# A = [0.1, 0.2, 0.1]
# B = [0.2, 0.1, 0.1]
# C = [0.2, 0.1, 0.2]

# print("A>B")
# optimizer_1.observe_behaivor_estimate(A, [B, C])
# optimizer_2.observe_behaivor_estimate(A, [B, C])
# optimizer_3.observe_behaivor_estimate(A, [C])
# optimizer_3.observe_behaivor_estimate(B, [C])
# print(" ")
# print("B>C")
# optimizer_3.observe_behaivor_estimate(B, [C])

# print(" ")
# print("C>A")

# optimizer_1.observe_behaivor_estimate(C, [A])
# optimizer_1.observe_behaivor_estimate(B, [A])
# optimizer_2.observe_behaivor_estimate(C, [A])
# optimizer_2.observe_behaivor_estimate(A, [A])
# optimizer_3.observe_behaivor_estimate(C, [A,B])

# print(optimizer_1.observed_x)
# print(optimizer_1.D)
# print(optimizer_2.observed_x)
# print(optimizer_2.D)
# print(optimizer_3.observed_x)
# print(optimizer_3.D)

# new_candidates = optimizer.optimize_acqf_and_get_observation()
# print("new",new_candidates)
# chosen_1 = new_candidates[0].tolist()
# others_2 = [new_candidates[1].tolist(), new_candidates[2].tolist()]
# optimizer.observe_behaivor_estimate(chosen_1, others_2)
# new_candidates_2 = optimizer.optimize_acqf_and_get_observation()
# print("new",new_candidates_2)
