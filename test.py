import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, dim, min_limit, max_limit):
        self.position = np.random.uniform(min_limit, max_limit, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.pbest = self.position
        self.fitness = float('inf')


def objective_function(x):
    # Define your objective function here
    # For demonstration purposes, using a simple quadratic function
    return np.sum(x**2)


def visualize_pso(particles, gbest, iteration):
    plt.figure()
    plt.title(f'Iteration {iteration}')
    
    for particle in particles:
        plt.scatter(particle.position[0], particle.position[1], color='blue', marker='o')
    
    plt.scatter(gbest.position[0], gbest.position[1], color='red', marker='x', label='Global Best')
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()


def particle_swarm_optimization(dim, num_particles, max_iterations, min_limit, max_limit):
    particles = [Particle(dim, min_limit, max_limit) for _ in range(num_particles)]
    
    gbest = min(particles, key=lambda particle: particle.fitness)
    
    for iteration in range(max_iterations):
        for particle in particles:
            particle.fitness = objective_function(particle.position)
            
            if particle.fitness < objective_function(particle.pbest):
                particle.pbest = particle.position
            
            if particle.fitness < objective_function(gbest.position):
                gbest = particle
        
        for particle in particles:
            # PSO update formula
            inertia_weight = 0.5
            c1, c2 = 1.5, 1.5
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            
            particle.velocity = (inertia_weight * particle.velocity +
                                 c1 * r1 * (particle.pbest - particle.position) +
                                 c2 * r2 * (gbest.position - particle.position))
            
            particle.position += particle.velocity
        
        visualize_pso(particles, gbest, iteration)

    print("Optimal solution:", gbest.position)
    print("Objective value at optimal solution:", objective_function(gbest.position))


# Example usage:
particle_swarm_optimization(dim=2, num_particles=10, max_iterations=20, min_limit=-5, max_limit=5)
