import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing


def show_terrain(base):
    light_blue = [0, 191, 255]
    blue = [65, 105, 225]
    green = [34, 139, 34]
    dark_green = [0, 100, 0]
    sandy = [210, 180, 140]
    beach = [238, 214, 175]
    snow = [255, 250, 250]
    mountain = [139, 137, 137]

    threshold = 0
    base = (base - np.amin(base)) / (np.amax(base) - np.amin(base))
    # print(f"Maximum value of base is {np.amax(base)}\nMinimum value of base is {np.amin(base)}")
    terrain = np.zeros(shape=(fidelity, fidelity, 3), dtype=np.uint8)
    terrain[np.where(base < 0.5)] = blue
    terrain[np.where((base > 0.5) & (base < 0.51))] = sandy
    terrain[np.where((base > 0.51) & (base < 0.55))] = beach
    terrain[np.where((base > 0.55) & (base < 0.65))] = green
    terrain[np.where((base > 0.65) & (base < 0.8))] = dark_green
    terrain[np.where((base > 0.8) & (base < 0.9))] = mountain
    terrain[np.where(base > 0.9)] = snow

    plt.imshow(terrain)
    plt.show()


def f_brownian(base, num_octaves=5, decay_factor=0.5):
    """
    Fractal Brownian motion. Adds more detail to the map
    :param decay_factor: The factor by which the next octaves' contributions are scaled - float in (0, 1)
    :param num_octaves: How many (sequential) higher octaves we want to add
    :param base: The original image generated using vanilla Perlin noise
    :return: The image with higher octaves added to it
    """
    output = np.copy(base)
    for idx in range(num_octaves):
        # We want to start with the second octave, not the first
        next_octave = get_octave(base, idx + 2)
        output += next_octave * np.power(decay_factor, idx + 1)
    return output


def get_octave(base, octave):
    larger = np.zeros(shape=(fidelity * octave, fidelity * octave))
    lr_flip = np.fliplr(base)
    ud_flip = np.flipud(base)
    all_flip = np.fliplr(base)
    all_flip = np.flipud(all_flip)
    for i in range(octave):
        for j in range(octave):
            if i % 2 == 0 and j % 2 == 0:
                larger[(i * fidelity):((i + 1) * fidelity), (j * fidelity):((j + 1) * fidelity)] = base
            if i % 2 == 1 and j % 2 == 0:
                larger[(i * fidelity):((i + 1) * fidelity), (j * fidelity):((j + 1) * fidelity)] = ud_flip
            if i % 2 == 0 and j % 2 == 1:
                larger[(i * fidelity):((i + 1) * fidelity), (j * fidelity):((j + 1) * fidelity)] = lr_flip
            if i % 2 == 1 and j % 2 == 1:
                larger[(i * fidelity):((i + 1) * fidelity), (j * fidelity):((j + 1) * fidelity)] = all_flip

    # Reshaping it into (batch_size=1, original shape, num_channels=1)
    larger = np.reshape(larger, newshape=(1, *np.shape(larger), 1))
    # Keras has pre-built layers that take the average over a window size (could experiment with maxpool too)
    mean_pool = tf.keras.layers.MaxPooling2D(pool_size=(octave, octave))
    smaller = mean_pool(larger)
    # print(f"smaller has size {np.shape(smaller)}")
    """
    # Bumping up the number of copies of the original
    larger = np.tile(base, (octave, octave))
    """
    return smaller[0, :, :, 0]


def fade(t):
    # return t * t * (3 - 2 * t)
    # This one has the second and first derivatives equal to zero at x=0 and 1
    return t * t * t * (t * (6 * t - 15) + 10)


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
    """
    Choose a point on a circle (2 Dimensions) uniformly. As we learned the hard way, distributions in higher dimensions
    don't play nicely
    :return: A unit 2D vector uniformly chosen from the circle
    """
    v = np.random.normal(0, 1, 2)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    np.random.seed(0)
    # I'd like for these to be global variables, hence no main() function
    dim = 8
    num_pixels = 64
    fidelity = num_pixels * (dim - 1)
    # Grid of random two-dimensional unit vectors
    grid = np.zeros((dim, dim, 2))
    image = np.zeros((fidelity, fidelity))
    """
    This defines the random vectors at the corners of the grid. Four adjacent random vectors form a square.
    Points inside a square will be dotted with vectors associated with the four points of the square
    """
    for i_idx in range(dim):
        for j_idx in range(dim):
            grid[i_idx, j_idx, :] = get_random_vec()

    """
    Iterate over all points in the image. For every pixel, we see which square it is in and then dot it with the
    appropriate vectors at the corners of the square to create a random image
    """
    for i_idx in range(fidelity):
        for j_idx in range(fidelity):
            # Offsets from grid points
            left_bot_off = np.array([i_idx % num_pixels, j_idx % num_pixels]) / num_pixels
            left_top_off = np.array([i_idx % num_pixels, j_idx % num_pixels - num_pixels]) / num_pixels
            right_bot_off = np.array([i_idx % num_pixels - num_pixels, j_idx % num_pixels]) / num_pixels
            right_top_off = np.array([i_idx % num_pixels - num_pixels, j_idx % num_pixels - num_pixels]) / num_pixels

            # Coordinates of the grid point to the bottom left of us
            x = np.floor(i_idx / num_pixels).astype(np.int32)
            y = np.floor(j_idx / num_pixels).astype(np.int32)
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
                x_fade = fade(left_bot_off[0])
                y_fade = fade(left_bot_off[1])
                # (Faded) linear interpolation for the left and right sides
                l_infl = lerp(left_bot_infl, left_top_infl, y_fade)
                r_infl = lerp(right_bot_infl, right_top_infl, y_fade)
                image[i_idx, j_idx] = lerp(l_infl, r_infl, x_fade)
            elif simple_sum:
                image[i_idx, j_idx] = left_bot_infl + left_top_infl + right_bot_infl + right_top_infl
            elif closest_grid:
                distances = np.array(
                    [np.linalg.norm(left_bot_off), np.linalg.norm(left_top_off), np.linalg.norm(right_bot_off),
                     np.linalg.norm(right_top_off)])
                if np.argmin(distances) == 0:
                    image[i_idx, j_idx] = left_bot_infl
                elif np.argmin(distances) == 1:
                    image[i_idx, j_idx] = left_top_infl
                elif np.argmin(distances) == 2:
                    image[i_idx, j_idx] = right_bot_infl
                elif np.argmin(distances) == 3:
                    image[i_idx, j_idx] = right_top_infl

    plt.imshow(image)
    plt.colorbar()
    plt.show()

    # octave = get_octave(image, 4)
    # plt.imshow(octave)
    # plt.colorbar()
    # plt.show()

    show_terrain(image)

    """
    Now we try to implement fractal Brownian Motion
    """
    # fractal_image = f_brownian(image, num_octaves=3, decay_factor=0.5)
    # plt.imshow(fractal_image)
    # plt.colorbar()
    # plt.show()
    #
    # show_terrain(fractal_image)
