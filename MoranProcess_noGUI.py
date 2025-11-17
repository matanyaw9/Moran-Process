import random

class MoranProcess:
    def __init__(self, N=50, freq_A=0.5, fitness_A=1.0, fitness_B=1.0):
        """
        N          - population size (fixed)
        freq_A     - initial frequency of type A (0-1)
        fitness_A  - relative fitness of A
        fitness_B  - relative fitness of B
        """
        self.N = N
        self.fitness_A = fitness_A
        self.fitness_B = fitness_B

        n_A = int(round(freq_A * N))
        n_B = N - n_A
        self.state = ['A'] * n_A + ['B'] * n_B
        random.shuffle(self.state)

    def step(self):
        """Perform one Moran step: reproduction + death."""
        count_A = self.state.count('A')
        if count_A == 0 or count_A == self.N:
            # Absorbing state: nothing happens
            return True

        # Choose reproducer with probability proportional to fitness
        weights = [
            self.fitness_A if t == 'A' else self.fitness_B
            for t in self.state
        ]
        total_w = sum(weights)
        r = random.uniform(0, total_w)
        cum = 0.0
        parent_type = None
        for t, w in zip(self.state, weights):
            cum += w
            if r <= cum:
                parent_type = t
                break

        # Choose random individual to die
        victim_idx = random.randrange(self.N)

        # Replace victim with offspring of parent
        self.state[victim_idx] = parent_type

    def freq_A(self):
        return self.state.count('A') / self.N

    def run_until_absorption(self, max_steps=1000, do_print=False):
        for step in range(max_steps):
            is_done = self.step()
            print(step, self.freq_A()) if do_print else None
            if is_done:
                return self.freq_A(), step
                
        return self.freq_A(), max_steps

if __name__ == "__main__":
    num_experiments = 10
    for _ in range(num_experiments):
        mp = MoranProcess(N=20, freq_A=0.5, fitness_A=1.0, fitness_B=1.0)
        final_freq, steps = mp.run_until_absorption(do_print=False)
        print(f"Final frequency of A: {final_freq} after {steps} steps")