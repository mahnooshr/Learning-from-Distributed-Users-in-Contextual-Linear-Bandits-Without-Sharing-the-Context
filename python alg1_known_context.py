import numpy as np
from typing import Dict, List, Tuple, Callable
from abc import ABC, abstractmethod

class StochasticQuantizer:
   
    
    @staticmethod
    def quantize(x: float, levels: int) -> int:
        """
        Quantize x ∈ [0, levels] to integer output
        Returns integer with E[output|x] = x
        """
        floor_x = int(np.floor(x))
        ceil_x = int(np.ceil(x))
        
        # Ensure within bounds
        floor_x = max(0, min(floor_x, levels))
        ceil_x = max(0, min(ceil_x, levels))
        
        # Probability of rounding up
        if floor_x == ceil_x:
            return floor_x
        
        p_up = x - floor_x
        if np.random.random() < p_up:
            return ceil_x
        else:
            return floor_x
    
    @staticmethod
    def quantize_interval(x: float, a: float, b: float, levels: int) -> float:
        """Quantize x ∈ [a, b] using levels"""
        # Scale to [0, levels]
        scaled = levels * (x - a) / (b - a)
        quantized = StochasticQuantizer.quantize(scaled, levels)
        # Scale back
        return a + quantized * (b - a) / levels

class ContextDistribution(ABC):
    """Abstract base class for context distributions"""
    
    @abstractmethod
    def sample(self) -> np.ndarray:
        """Sample a context vector"""
        pass
    
    @abstractmethod
    def expected_argmax(self, theta: np.ndarray, action_contexts: Dict[int, 'ContextDistribution']) -> np.ndarray:
        """Compute E[argmax_a <X_a, theta>]"""
        pass

class GaussianContextDistribution(ContextDistribution):
    """Gaussian context distribution"""
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov
        self.d = len(mean)
    
    def sample(self) -> np.ndarray:
        return np.random.multivariate_normal(self.mean, self.cov)
    
    def expected_argmax(self, theta: np.ndarray, action_contexts: Dict[int, 'ContextDistribution']) -> np.ndarray:
        """
        For Gaussian distributions, this requires numerical integration
        or Monte Carlo approximation
        """
        n_samples = 10000
        sum_argmax = np.zeros(self.d)
        
        for _ in range(n_samples):
            # Sample contexts for all actions
            contexts = {}
            for a, dist in action_contexts.items():
                contexts[a] = dist.sample()
            
            # Find best action
            best_a = max(contexts.keys(), key=lambda a: np.dot(contexts[a], theta))
            sum_argmax += contexts[best_a]
        
        return sum_argmax / n_samples

class Algorithm1KnownDistribution:
    """Algorithm 1: Known context distribution (0 bits for context)"""
    
    def __init__(self, d: int, action_set: List[int], context_distributions: Dict[int, ContextDistribution]):
        self.d = d
        self.action_set = action_set
        self.context_distributions = context_distributions
        
        # Precompute X* mapping
        self.theta_space = self._create_theta_space()
        self.X_star_mapping = self._precompute_X_star()
        self.X_set = list(self.X_star_mapping.values())
        
        # Initialize LinUCB for single context problem
        self.V = np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
    def _create_theta_space(self) -> List[np.ndarray]:
        """Create discretized theta space"""
        # Simple grid discretization
        n_points = 10  # Per dimension
        theta_space = []
        
        # Create grid points
        grid_1d = np.linspace(-1, 1, n_points)
        
        # For simplicity, use random sampling instead of full grid
        for _ in range(100):
            theta = np.random.uniform(-1, 1, self.d)
            theta = theta / np.linalg.norm(theta)  # Normalize
            theta_space.append(theta)
        
        return theta_space
    
    def _precompute_X_star(self) -> Dict[int, np.ndarray]:
        """Precompute X*(theta) for all theta in discretized space"""
        X_star_mapping = {}
        
        for i, theta in enumerate(self.theta_space):
            X_star = self._compute_X_star(theta)
            X_star_mapping[i] = X_star
        
        return X_star_mapping
    
    def _compute_X_star(self, theta: np.ndarray) -> np.ndarray:
        """Compute X*(theta) = E[argmax_a <X_a, theta>]"""
        return self.context_distributions[self.action_set[0]].expected_argmax(
            theta, self.context_distributions
        )
    
    def _find_inverse_X_star(self, x: np.ndarray) -> np.ndarray:
        """Find theta such that X*(theta) ≈ x"""
        best_idx = min(range(len(self.X_star_mapping)), 
                      key=lambda i: np.linalg.norm(self.X_star_mapping[i] - x))
        return self.theta_space[best_idx]
    
    def select_action_central(self) -> Tuple[np.ndarray, np.ndarray]:
        """Central learner selects action using LinUCB on X set"""
        # LinUCB exploration
        alpha = 1.0
        
        best_ucb = -np.inf
        best_x = None
        
        for x in self.X_set:
            theta_est = np.linalg.solve(self.V, self.b)
            x_norm = np.sqrt(x.T @ np.linalg.solve(self.V, x))
            ucb = x @ theta_est + alpha * x_norm
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_x = x
        
        # Convert to theta for agent
        theta_to_send = self._find_inverse_X_star(best_x)
        
        return best_x, theta_to_send
    
    def agent_play(self, theta_hat: np.ndarray, contexts: Dict[int, np.ndarray]) -> Tuple[int, float]:
        """Agent plays best action given theta_hat"""
        best_action = max(self.action_set, 
                         key=lambda a: np.dot(contexts[a], theta_hat))
        
        # Compute reward (with noise)
        true_theta = np.ones(self.d) / np.sqrt(self.d)  # Example true parameter
        reward = np.dot(contexts[best_action], true_theta) + 0.1 * np.random.randn()
        reward = np.clip(reward, 0, 1)
        
        return best_action, reward
    
    def update_central(self, x: np.ndarray, reward_quantized: float):
        """Update LinUCB with quantized reward"""
        self.V += np.outer(x, x)
        self.b += reward_quantized * x
        self.theta_hat = np.linalg.solve(self.V, self.b)
    
    def run_episode(self) -> float:
        """Run one episode and return regret"""
        # Central learner selects
        x_selected, theta_to_send = self.select_action_central()
        
        # Agent observes contexts and plays
        contexts = {a: self.context_distributions[a].sample() 
                   for a in self.action_set}
        action_played, reward = self.agent_play(theta_to_send, contexts)
        
        # Quantize reward (1 bit)
        reward_quantized = StochasticQuantizer.quantize(reward, 1)
        
        # Update central learner
        self.update_central(x_selected, reward_quantized)
        
        # Compute regret
        true_theta = np.ones(self.d) / np.sqrt(self.d)
        best_reward = max(np.dot(contexts[a], true_theta) for a in self.action_set)
        regret = best_reward - reward
        
        return regret

class Algorithm2UnknownDistribution:
    """Algorithm 2: Unknown context distribution (~5d bits per context)"""
    
    def __init__(self, d: int, action_set: List[int]):
        self.d = d
        self.action_set = action_set
        self.m = int(np.ceil(np.sqrt(d)))
        
        # Initialize parameters
        self.theta_hat = np.zeros(d)
        self.V_tilde = np.eye(d) * 0.01  # Small regularization
        self.u = np.zeros(d)
        
        # For enumeration of quantized vectors
        self.max_norm = 2 * d
        
    def _quantize_context(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize context vector x"""
        # Get signs (d bits)
        signs = np.sign(x)
        signs[signs == 0] = 1
        
        # Quantize magnitude (log(|Q|) bits)
        x_scaled = self.m * np.abs(x)
        x_quantized = np.array([StochasticQuantizer.quantize(xi, self.m) 
                               for xi in x_scaled])
        
        # Compute reconstruction
        x_hat = signs * x_quantized / self.m
        
        # Quantize diagonal correction (d bits)
        x_squared = x * x
        x_hat_squared = x_hat * x_hat
        diff = x_squared - x_hat_squared
        
        # Each difference is in [-3/m, 3/m]
        e_squared = np.array([StochasticQuantizer.quantize_interval(d, -3/self.m, 3/self.m, 1) 
                             for d in diff])
        
        return signs, x_quantized, e_squared
    
    def send_theta_to_agent(self) -> np.ndarray:
        """Send current theta estimate to agent"""
        return self.theta_hat.copy()
    
    def agent_play(self, theta_hat: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        Agent observes contexts, plays best action, and returns quantized data
        """
        # Sample contexts
        contexts = {}
        for a in self.action_set:
            # Random context generation for demo
            x = np.random.randn(self.d)
            x = x / np.linalg.norm(x)  # Normalize
            contexts[a] = x
        
        # Play best action
        best_action = max(self.action_set, 
                         key=lambda a: np.dot(contexts[a], theta_hat))
        
        # Get reward
        true_theta = np.ones(self.d) / np.sqrt(self.d)  # Example true parameter
        reward = np.dot(contexts[best_action], true_theta) + 0.1 * np.random.randn()
        reward = np.clip(reward, 0, 1)
        
        return contexts[best_action], best_action, reward
    
    def receive_and_update(self, signs: np.ndarray, x_quantized: np.ndarray, 
                          e_squared: np.ndarray, reward_quantized: float):
        """Central learner receives quantized data and updates estimates"""
        # Reconstruct context estimate
        x_hat = signs * x_quantized / self.m
        x_hat_diagonal = x_hat * x_hat + e_squared
        
        # Update statistics
        self.u += reward_quantized * x_hat
        
        # Update V_tilde
        outer_prod = np.outer(x_hat, x_hat)
        self.V_tilde += outer_prod - np.diag(np.diag(outer_prod)) + np.diag(x_hat_diagonal)
        
        # Update theta estimate
        try:
            self.theta_hat = np.linalg.solve(self.V_tilde, self.u)
        except:
            self.theta_hat = np.linalg.pinv(self.V_tilde) @ self.u
    
    def run_episode(self) -> float:
        """Run one episode with ~5d bits communication"""
        # Send theta to agent
        theta_to_send = self.send_theta_to_agent()
        
        # Agent plays
        context_played, action_played, reward = self.agent_play(theta_to_send)
        
        # Quantize context (d + log(|Q|) + d bits)
        signs, x_quantized, e_squared = self._quantize_context(context_played)
        
        # Quantize reward (1 bit)
        reward_quantized = StochasticQuantizer.quantize(reward, 1)
        
        # Update central learner
        self.receive_and_update(signs, x_quantized, e_squared, reward_quantized)
        
        # Compute regret
        true_theta = np.ones(self.d) / np.sqrt(self.d)
        best_possible = context_played @ true_theta  # Simplified for demo
        regret = best_possible - reward
        
        return regret

# Example usage
def run_experiments():
   
    d = 10  # Dimension
    n_actions = 5
    action_set = list(range(n_actions))
    T = 1000  # Time horizon
    
    # Create context distributions for Algorithm 1
    context_dists = {}
    for a in action_set:
        mean = np.random.randn(d) * 0.5
        cov = np.eye(d) * 0.1
        context_dists[a] = GaussianContextDistribution(mean, cov)
    
    # Run Algorithm 1
    print("Running Algorithm 1 (Known Distribution)...")
    alg1 = Algorithm1KnownDistribution(d, action_set, context_dists)
    regret1 = 0
    
    for t in range(T):
        regret1 += alg1.run_episode()
        if (t + 1) % 100 == 0:
            print(f"  Step {t+1}: Cumulative regret = {regret1:.2f}")
    
    # Run Algorithm 2
    print("\nRunning Algorithm 2 (Unknown Distribution)...")
    alg2 = Algorithm2UnknownDistribution(d, action_set)
    regret2 = 0
    
    for t in range(T):
        regret2 += alg2.run_episode()
        if (t + 1) % 100 == 0:
            print(f"  Step {t+1}: Cumulative regret = {regret2:.2f}")
    
    print(f"\nFinal Results:")
    print(f"Algorithm 1 total regret: {regret1:.2f}")
    print(f"Algorithm 2 total regret: {regret2:.2f}")
    print(f"Communication cost per round:")
    print(f"  Algorithm 1: 1 bit")
    print(f"  Algorithm 2: ~{5*d} bits")

if __name__ == "__main__":
    np.random.seed(42)
    run_experiments()