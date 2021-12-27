import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def fade(t):
    # return t * t * t * (t * (6 * t - 15) + 10)
    return t * t * (3 - 2 * t)


def lerp(a, b, dist):
    """
    Linear interpolation
    :param a: Dot product at the first grid point of interest
    :param b: Dot product at the second grid point of interest
    :param dist: Distance from the first grid point of interest (or faded distance)
    :return: The linearly interpolated result
    """
    # Equivalent to a(1-x)+bx
    return a + dist * (b - a)


def get_random_vec():
    v = np.random.normal(0, 1, 2)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    np.random.seed(0)
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

    for i in range(fidelity):
        for j in range(fidelity):
            # Offsets from grid points
            left_bot_off = np.array([i % num_pixels, j % num_pixels])
            left_top_off = np.array([i % num_pixels, j % num_pixels - num_pixels])
            right_bot_off = np.array([i % num_pixels - num_pixels, j % num_pixels])
            right_top_off = np.array([i % num_pixels - num_pixels, j % num_pixels - num_pixels])

            # Coordinates of the grid point to the bottom left of us
            x = np.floor(i / num_pixels).astype(np.int32)
            y = np.floor(j / num_pixels).astype(np.int32)
            # Influence values
            left_bot_infl = np.dot(left_bot_off, grid[x, y, :])
            left_top_infl = np.dot(left_top_off, grid[x, y + 1, :])
            right_bot_infl = np.dot(right_bot_off, grid[x + 1, y, :])
            right_top_infl = np.dot(right_top_off, grid[x + 1, y + 1, :])
            """
            This is where we should be using a fade function. Just testing that it works, then will change it
            Calculate dropoff values and then sum them up
            """
            faded = True
            simple_sum = False
            closest_grid = False
            if faded:
                x_fade = fade(left_bot_off[0] / num_pixels)
                y_fade = fade(left_bot_off[1] / num_pixels)
                # (Faded) linear interpolation for the left and right sides
                l_infl = lerp(left_bot_infl, left_top_infl, y_fade)
                r_infl = lerp(right_bot_infl, right_top_infl, y_fade)
                image[i, j] = lerp(l_infl, r_infl, x_fade)
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
    plt.colorbar()
    plt.show()
    # plt.hist(image[1, :], density=True)
    # plt.show()
