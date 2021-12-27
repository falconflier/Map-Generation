import numpy as np
import matplotlib.pyplot as plt


def get_increase_prob(grid, pos, proposed):
    # the amount of "randomness" we want in generation
    temp = 4
    dim = np.size(grid, axis=0)
    num_diff = 0
    # Iterating over nearest neighbors (including diagonal neighbors)
    for i in range(3):
        x = pos[0] + i - 1
        if x < 0 or x >= dim:
            continue
        for j in range(3):
            y = pos[1] + j - 1
            if y < 0 or y >= dim:
                continue
            # If we're here, have avoided array-out-of-bounds errors and can start counting terrain
            if grid[x, y] != proposed:
                num_diff += 1
    prob = np.exp(- num_diff / temp)
    return prob


def main():
    dim = 32
    # The percentage and absolute number of squares in the grid that we want to be land
    perc_land = 0.33
    tar_land = int(dim * dim * perc_land)
    # The actual amount of land we currently have
    ac_land = 0
    # The number of continents we want
    num_continents = 2
    # The grid where we start building our continents
    grid = np.zeros(shape=[dim, dim])

    # start1 = np.random.randint(0, dim, size=2)
    start1 = np.array([16, 16])
    grid[start1[0], start1[1]] = 1
    # start2 = start1 + dim // 2
    # start2 = start2 % dim
    # grid[start2[0], start2[1]] = 2

    for i in range(100):
        step1 = np.random.normal(scale=4, size=2).astype(np.int32)
        prop1 = start1 + step1
        prop1 = prop1 // dim

        if grid[prop1[0], prop1[1]] == 0:
            if ac_land < tar_land \
                    or np.random.rand() < get_increase_prob(grid, prop1, 1):  # Stochastic if adding more land than target
                grid[prop1[0], prop1[1]] = 1
                print("More land added")
        else:
            print("Hit land")

        # step2 = np.random.normal(scale=step_size, size=2).astype(np.int32)
        # start2 += step2
        # start2 = start2 // dim

    plt.imshow(grid)
    plt.show()


if __name__ == "__main__":
    main()
