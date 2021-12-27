import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def fade(t):
    return t * t * t * (t * (6 * t - 15) + 10)


def get_random_vec():
    v = np.random.normal(0, 1, 2)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    # I'd like for these to be global variables, hence no main() function
    dim = 8
    num_pixels = 16
    fidelity = num_pixels * (dim - 1)
    # Grid of random two-dimensional unit vectors
    grid = np.zeros((dim, dim, 2))
    image = np.zeros((fidelity, fidelity))
    for i in range(dim):
        for j in range(dim):
            grid[i, j, :] = get_random_vec()

    # plt.imshow(grid[:, :, 0])
    # plt.show()
    # plt.imshow(grid[:, :, 1])
    # plt.show()
    for i in range(fidelity):
        for j in range(fidelity):
            # Offsets from grid points
            left_bot_off = np.array([i % num_pixels, j % num_pixels])
            left_top_off = np.array([i % num_pixels, j % num_pixels - num_pixels])
            right_bot_off = np.array([i % num_pixels - num_pixels, j % num_pixels])
            right_top_off = np.array([i % num_pixels - num_pixels, j % num_pixels - num_pixels])
            # If we're at one of the corners, then the contribution is 0
            if np.linalg.norm(left_bot_off) == 0:
                left_bot_off = np.zeros(2)
            else:
                left_bot_off = left_bot_off / np.linalg.norm(left_bot_off)
            if np.linalg.norm(left_top_off) == 0:
                left_top_off = np.zeros(2)
            else:
                left_top_off = left_top_off / np.linalg.norm(left_top_off)
            if np.linalg.norm(right_bot_off) == 0:
                right_bot_off = np.zeros(2)
            else:
                right_bot_off = right_bot_off / np.linalg.norm(right_bot_off)
            if np.linalg.norm(right_top_off) == 0:
                right_top_off = np.zeros(2)
            else:
                right_top_off = right_top_off / np.linalg.norm(right_top_off)

            # Coordinates of the grid point to the bottom left of us
            x = np.floor(i / num_pixels).astype(np.int32)
            y = np.floor(j / num_pixels).astype(np.int32)
            # print(f"{x}, {y}")
            # Influence values
            left_bot_infl = np.dot(left_bot_off, grid[x, y, :])
            left_top_infl = np.dot(left_top_off, grid[x, y + 1, :])
            right_bot_infl = np.dot(right_bot_off, grid[x + 1, y, :])
            right_top_infl = np.dot(right_top_off, grid[x + 1, y + 1, :])
            """
            This is where we should be using a fade function. Just testing that it works, then will change it
            Calculate dropoff values and then sum them up
            """
            faded = False
            simple_sum = False
            closest_grid = True
            if faded:
                left_bot_drop = fade(1 - np.abs(left_bot_off[0])) * fade(1 - np.abs(left_bot_off[1]))
                left_top_drop = fade(1 - np.abs(left_top_off[0])) * fade(1 - np.abs(left_top_off[1]))
                right_bot_drop = fade(1 - np.abs(right_bot_off[0])) * fade(1 - np.abs(right_bot_off[1]))
                right_top_drop = fade(1 - np.abs(right_top_off[0])) * fade(1 - np.abs(right_top_off[1]))
                image[i, j] += left_bot_drop * left_bot_infl
                image[i, j] += left_top_drop * left_top_infl
                image[i, j] += right_bot_drop * right_bot_infl
                image[i, j] += right_top_drop * right_top_infl
            elif simple_sum:
                image[i, j] = left_bot_infl + left_top_infl + right_bot_infl + right_top_infl
            elif closest_grid:
                distances = np.array([np.linalg.norm(left_bot_off), np.linalg.norm(left_top_off), np.linalg.norm(right_bot_off), np.linalg.norm(right_top_off)])
                if np.argmin(distances) == 0:
                    image[i, j] = left_bot_infl
                elif np.argmin(distances) == 1:
                    image[i, j] = left_top_infl
                elif np.argmin(distances) == 2:
                    image[i, j] = right_bot_infl
                elif np.argmin(distances) == 3:
                    image[i, j] = right_top_infl

    plt.imshow(image)
    plt.show()
    plt.hist(image[1, :], density=True)
    plt.show()
