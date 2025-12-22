import numpy as np




def run_moran_process(n, num_steps):
    population = np.zeros(n, dtype=int)
    population[0] = 1  # Start with one individual of type 1
    prob_matrix = np.full(shape=(n, n), fill_value=1/n)
    for step in range(num_steps):
        # Select an individual to reproduce based on the probability matrix
        reproducer = np.random.choice(n, p=prob_matrix.sum(axis=1) / prob_matrix.sum())
        # Select an individual to be replaced
        replacee = np.random.choice(n, p=prob_matrix[reproducer] / prob_matrix[reproducer].sum())
        # Update population
        population[replacee] = population[reproducer]
        if population.sum() == n or population.sum() == 0:
            result = population[0]
            print(f"Fixation reached at step {step}\t color: {population[0]}")
            break
    else:
        print("No fixation reached. 0: ", np.sum(population == 0), " 1: ", np.sum(population == 1))
        result = -1
    


    return result, step


def main():
    num_experiments = 100
    n = 10
    num_steps = 500
    zero_wins, one_wins, draws = 0, 0, 0
    steps_taken = []
    for experiment in range(num_experiments):
        print(f"Experiment {experiment + 1}")
        res, step = run_moran_process(n, num_steps)
        if res == 0:
            zero_wins += 1
        elif res == 1:
            one_wins += 1
        else:
            draws += 1
        steps_taken.append(step)

    print("\nAll experiments completed.\n")
    print(f"Zero wins: {zero_wins}")
    print(f"One wins: {one_wins}")
    print(f"Draws: {draws}")
    print(f"Average steps taken: {np.mean(steps_taken)}")



if __name__ == "__main__":
    main()
