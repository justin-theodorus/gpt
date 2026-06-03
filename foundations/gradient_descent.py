class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        objective = lambda x: x**2
        derivative = lambda x: 2*x
        x = init
        for _ in range(iterations):
            x -= learning_rate * derivative(x)
        
        return round(x, 5)
