import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Particle:
    def __init__(self, dim, min_limit, max_limit):
        self.position = np.random.uniform(min_limit, max_limit, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.pbest = self.position
        self.fitness = float('inf')


def objective_function(x):
    return np.sum(x**2)


def update_plot(frame, particles, gbest, sc, ax, dim, text_optimal_solution, text_objective_value):
    for particle in particles:
        particle.fitness = objective_function(particle.position)

        if particle.fitness < objective_function(particle.pbest):
            particle.pbest = particle.position

        if particle.fitness < objective_function(gbest.position):
            gbest = particle

    for particle in particles:
        inertia_weight = 0.5
        c1, c2 = 1.5, 1.5
        r1, r2 = np.random.rand(dim), np.random.rand(dim)

        particle.velocity = (inertia_weight * particle.velocity +
                             c1 * r1 * (particle.pbest - particle.position) +
                             c2 * r2 * (gbest.position - particle.position))

        particle.position += particle.velocity

    sc.set_offsets(np.array([particle.position for particle in particles]))

    ax.set_title(f'Iteration {frame}')

    # Update optimal solution and objective value text
    text_optimal_solution.set_text(f'Optimal Solution: {gbest.position}')
    text_objective_value.set_text(f'Objective Value: {objective_function(gbest.position)}')

    return sc, text_optimal_solution, text_objective_value


def particle_swarm_optimization_animation(dim, num_particles, max_iterations, min_limit, max_limit, loop_duration=5):
    particles = [Particle(dim, min_limit, max_limit) for _ in range(num_particles)]
    gbest = min(particles, key=lambda particle: particle.fitness)

    fig, ax = plt.subplots()
    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)
    sc = ax.scatter([particle.position[0] for particle in particles],
                    [particle.position[1] for particle in particles],
                    color='blue', marker='o')

    # Initial text annotations
    text_optimal_solution = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    text_objective_value = ax.text(0.05, 0.85, '', transform=ax.transAxes)

    frames_per_second = 24  # Adjust as needed
    initial_frames = 10 * frames_per_second
    loop_frames = max_iterations * frames_per_second - initial_frames
    interval = 1000 / frames_per_second  # Interval in milliseconds

    # Use min(initial_frames, 1) to ensure at least one frame for the initial animation
    anim_initial = FuncAnimation(fig, update_plot, frames=min(initial_frames, 1),
                                 fargs=(particles, gbest, sc, ax, dim, text_optimal_solution, text_objective_value),
                                 interval=interval, blit=False)

    anim_loop = FuncAnimation(fig, update_plot, frames=loop_frames,
                              fargs=(particles, gbest, sc, ax, dim, text_optimal_solution, text_objective_value),
                              interval=interval, blit=False)

    # Combine animations to run the loop after 5 seconds
    html = f"{anim_initial.to_jshtml()}<script>setTimeout(() => {anim_loop.to_jshtml()}, {loop_duration * 1000})</script>"

    # Save the HTML to a file (optional)
    with open("animation_output.html", "w") as html_file:
        html_file.write(html)

    # Display the animation in the default web browser
    import webbrowser
    webbrowser.open("animation_output.html")


# Example usage:
particle_swarm_optimization_animation(dim=2, num_particles=10, max_iterations=10, min_limit=-5, max_limit=5, loop_duration=5)
