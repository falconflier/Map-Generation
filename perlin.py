import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Color definitions. Can change these to whatever RGB values are desired
"""
color_dict = {"light_blue": [0, 191, 255], "blue": [65, 105, 225], "green": [34, 139, 34], "dark_green": [0, 100, 0],
              "sandy": [210, 180, 140], "beach": [238, 214, 175], "snow": [255, 250, 250], "mountain": [139, 137, 137],
              "purple": [200, 0, 200]}

"""
This dictionary defines the height range for different colors. It assumes min-maxing, but will work regardless of
scaling (just might be all water). Can change these to be anything you want, just remember to change both the upper
bound and lower bounds (otherwise you'll get overlap, or black bands)
"""
height_dict = {"light_blue": [0.45, 0.5], "blue": [- np.inf, 0.45], "green": [0.55, 0.65], "dark_green": [0.65, 0.8],
               "sandy": [0.5, 0.51], "beach": [0.51, 0.55], "snow": [0.9, np.inf], "mountain": [0.8, 0.9],
               "purple": [200, 0, 200]}


def show_terrain(base):
    threshold = 0
    # Making sure that it's Min-Max scaled
    base = (base - np.amin(base)) / (np.amax(base) - np.amin(base))
    # print(f"Maximum value of base is {np.amax(base)}\nMinimum value of base is {np.amin(base)}")
    terrain = np.zeros(shape=(fidelity, fidelity, 3), dtype=np.uint8)
    terrain[np.where((base >= height_dict["blue"][0]) & (base < height_dict["blue"][1]))] = color_dict["blue"]
    terrain[np.where((base >= height_dict["light_blue"][0]) & (base < height_dict["light_blue"][1]))] = color_dict[
        "light_blue"]
    terrain[np.where((base >= height_dict["sandy"][0]) & (base < height_dict["sandy"][1]))] = color_dict["sandy"]
    terrain[np.where((base >= height_dict["beach"][0]) & (base < height_dict["beach"][1]))] = color_dict["beach"]
    terrain[np.where((base >= height_dict["green"][0]) & (base < height_dict["green"][1]))] = color_dict["green"]
    terrain[np.where((base >= height_dict["dark_green"][0]) & (base < height_dict["dark_green"][1]))] = color_dict[
        "dark_green"]
    terrain[np.where((base >= height_dict["mountain"][0]) & (base < height_dict["mountain"][1]))] = color_dict[
        "mountain"]
    terrain[np.where((base >= height_dict["snow"][0]) & (base < height_dict["snow"][1]))] = color_dict["snow"]

    plt.imshow(terrain)
    plt.show()


def get_frequent_height(noise_grid):
    keys = color_dict.keys()
    frequency = dict(zip(keys, np.zeros(len(keys))))
    for key in keys:
        frequency[key] = np.count_nonzero(
            np.where((noise_grid >= height_dict[key][0]) & (noise_grid < height_dict[key][1])))

    """
    This works, somehow. Source:
    https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    """
    return color_dict[max(frequency, key=frequency.get)]


def show_large_downsample_image(base, partition_pixels=8):
    """
    Does much the same thing as show_downsample_image (and was written first) but keeps the size of the the output grid
    close to (fidelity, fidelity). This means that many pixels are redundant and show the same color
    :param base:
    :param partition_pixels:
    :return:
    """
    # Making sure that it's Min-Max scaled
    base = (base - np.amin(base)) / (np.amax(base) - np.amin(base))
    partitions = fidelity // partition_pixels
    terrain = - np.ones(shape=(partitions * partition_pixels, partitions * partition_pixels, 3), dtype=np.uint8)
    for i in range(partitions):
        for j in range(partitions):
            color_rgb = get_frequent_height(
                base[i * partition_pixels:(i + 1) * partition_pixels, j * partition_pixels:(j + 1) * partition_pixels])
            # print(f"Color rgb is {color_rgb}")
            terrain[i * partition_pixels:(i + 1) * partition_pixels, j * partition_pixels:(j + 1) * partition_pixels, :] = color_rgb
    """
    # This step drops any pixels that have not been filled (which will still contain purple as their value)
    # terrain != color_dict["purple"] checks where this isn't the case, and we use np.where to filter where this is
    # true; indices which we keep. This unfortunately leaves the array in a 1-D mess, so we have to reshape it.
    # Fortunately, we know that the remnants will be divisible by three, and we can take the square root to find the
    # new image shape. The fact that the square root will yield an integer is due to us starting with square images
    # and using the same number of partitions for both the x and y dimensions
    # """
    # terrain = terrain[np.where(terrain != -1)]
    # length = int(np.sqrt(np.size(terrain) / 3))
    # terrain = np.reshape(terrain, (length, length, 3)).astype(np.uint8)
    # Show the results
    plt.imshow(terrain)
    plt.show()


def show_downsample_image(base, partition_pixels=8):
    # Making sure that it's Min-Max scaled
    base = (base - np.amin(base)) / (np.amax(base) - np.amin(base))
    partitions = fidelity // partition_pixels
    terrain = - np.ones(shape=(partitions, partitions, 3), dtype=np.uint8)
    for i in range(partitions):
        for j in range(partitions):
            color_rgb = get_frequent_height(
                base[i * partition_pixels:(i + 1) * partition_pixels, j * partition_pixels:(j + 1) * partition_pixels])
            # print(f"Color rgb is {color_rgb}")
            terrain[i, j, :] = color_rgb
    # Show the results
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
    # We need a place to hold all of the extra pixels
    larger = np.zeros(shape=(fidelity * octave, fidelity * octave))
    # These are the flipped versions of the basic image we start with. We need this to avoid discontinuities
    lr_flip = np.fliplr(base)
    ud_flip = np.flipud(base)
    all_flip = np.fliplr(base)
    all_flip = np.flipud(all_flip)
    # This populates 'larger' with versions of 'base,' flipped appropriately to give the impression of continuity
    for i in range(octave):
        for j in range(octave):
            if i % 2 == 0 and j % 2 == 0:
                larger[(i * fidelity):((i + 1) * fidelity), (j * fidelity):((j + 1) * fidelity)] = base
            if i % 2 == 1 and j % 2 == 0:
                """
                Ngl I thought that this would be the lr_flip. But the grid doesn't work exactly how I thought,
                and this results in a nice octave
                """
                larger[(i * fidelity):((i + 1) * fidelity), (j * fidelity):((j + 1) * fidelity)] = ud_flip
            if i % 2 == 0 and j % 2 == 1:
                larger[(i * fidelity):((i + 1) * fidelity), (j * fidelity):((j + 1) * fidelity)] = lr_flip
            if i % 2 == 1 and j % 2 == 1:
                larger[(i * fidelity):((i + 1) * fidelity), (j * fidelity):((j + 1) * fidelity)] = all_flip

    # Reshaping it into (batch_size=1, original shape, num_channels=1) so that we can use tensorflow on it
    larger = np.reshape(larger, newshape=(1, *np.shape(larger), 1))
    # Keras has pre-built layers that take the average over a window size (could experiment with maxpool too)
    pool = tf.keras.layers.MaxPooling2D(pool_size=(octave, octave))
    smaller = pool(larger)
    # print(f"smaller has size {np.shape(smaller)}")
    """
    # An old version of what I did that leaves sharp discontinuities going across octaves
    larger = np.tile(base, (octave, octave))
    """
    return smaller[0, :, :, 0]


def fade(t):
    """
    Fade function. Used to prevent sharp edges during linear interpolation
    :param t: Distance to grid point of interest. Should be a float in [0, 1]
    :return: Faded distance from grid point of interest
    """
    # return t * t * (3 - 2 * t)
    """
    This is a more sophisticated fade function that has its second and first derivatives equal to zero at x=0 and 1.
    It's equivalent to 6t^5-15t^4+10t^3, but someone wrote it like this, I assume because it cuts down the number
    of operations required for computation
    """
    return t * t * t * (t * (6 * t - 15) + 10)


def lerp(a, b, dist):
    """
    Linear interpolation
    :param a: Dot product of displacement vector with the first grid point of interest
    :param b: Dot product of displacement vector with the second grid point of interest
    :param dist: Distance from the first grid point of interest (or faded distance)
    :return: The linearly interpolated result
    """
    # Equivalent to a(1-x)+bx
    return a + dist * (b - a)


def get_circular_filter():
    """
    Function that returns a mask.
    :return: A rotationally symmetric dropoff filter that can be multiplied with noise images
    """
    coord_dim = np.floor(fidelity / 2)
    # This accounts for the even or odd number of pixels in the image
    pad = fidelity % 2
    x, y = np.ogrid[-coord_dim:coord_dim + pad, -coord_dim:coord_dim + pad]
    r = np.sqrt(np.square(x) + np.square(y))

    delta_r = 1
    mask = np.zeros((fidelity, fidelity))
    for i in range(fidelity, 1, -delta_r):
        # print(i)
        # mask[np.where((r >= i - 1) & (r < i))] = 1 - i / fidelity
        mask[np.where((r < i))] = 1 - fade(i / fidelity)
    return mask


def get_random_vec():
    """
    Choose a point on a circle (2 Dimensions) uniformly. As we learned the hard way, distributions in higher dimensions
    don't play nicely
    :return: A unit 2D vector uniformly chosen from the circle
    """
    v = np.random.normal(0, 1, 2)
    return v / np.linalg.norm(v)


def perlin():
    """
    Implements the Perlin Noise algorithm
    :return: A perlin noise image (first octave only)
    """
    # Grid of random two-dimensional unit vectors
    grid = np.zeros((dim, dim, 2))
    noise_image = np.zeros((fidelity, fidelity))
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
            Exactly one of the next Booleans should be set to true
            faded=True implies that we'll use the proper (Perlin approved) method of interpolation
            simple_sum=True implies that we'll just add up the results (no linear interpolation at all)
            closest_grid=True implies that we'll only consider the contribution from the nearest grid point
            """
            faded = True
            simple_sum = False
            closest_grid = False
            if faded:
                # Faded x and y distances from the lower left grid point
                x_fade = fade(left_bot_off[0])
                y_fade = fade(left_bot_off[1])
                # (Faded) linear interpolation for the left and right sides
                l_infl = lerp(left_bot_infl, left_top_infl, y_fade)
                r_infl = lerp(right_bot_infl, right_top_infl, y_fade)
                # Combining it together to get the actual result
                noise_image[i_idx, j_idx] = lerp(l_infl, r_infl, x_fade)
            elif simple_sum:
                # Just add everything up. Looks terrible but is a decent sanity check
                noise_image[i_idx, j_idx] = left_bot_infl + left_top_infl + right_bot_infl + right_top_infl
            elif closest_grid:
                # Array that keeps track of the distance to each of the four closest grid points. ORDER MATTERS
                distances = np.array(
                    [np.linalg.norm(left_bot_off), np.linalg.norm(left_top_off), np.linalg.norm(right_bot_off),
                     np.linalg.norm(right_top_off)])
                """
                Go through different cases. I maybe could have used a dictionary, but then I wouldn't have been able
                to use numpy's functions to find the argmin, which lets me easily find the closest grid point.
                """
                if np.argmin(distances) == 0:
                    noise_image[i_idx, j_idx] = left_bot_infl
                elif np.argmin(distances) == 1:
                    noise_image[i_idx, j_idx] = left_top_infl
                elif np.argmin(distances) == 2:
                    noise_image[i_idx, j_idx] = right_bot_infl
                elif np.argmin(distances) == 3:
                    noise_image[i_idx, j_idx] = right_top_infl

    # Min max scaling it
    noise_image = (noise_image - np.amin(noise_image)) / (np.amax(noise_image) - np.amin(noise_image))
    return noise_image


def main():
    plt.imshow(image)
    plt.colorbar()
    plt.show()

    octave = get_octave(image, 4)
    plt.imshow(octave)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    # I'd like for these to be global variables, hence these are defined before the main() function
    dim = 8
    num_pixels = 64
    fidelity = num_pixels * (dim - 1)

    image = perlin()
    # show_terrain(image)

    """
    Implements fractal Brownian Motion
    """
    fractal_image = f_brownian(image, num_octaves=7, decay_factor=0.5)
    # show_terrain(fractal_image)

    """
    This turns it into a proper archipelago
    """
    circle_filter = get_circular_filter()
    archipelago_noise = fractal_image * circle_filter
    # plt.imshow(archipelago_noise)
    # plt.show()
    show_terrain(archipelago_noise)

    """
    Now we turn it into a polytopia-esque map
    """
    show_large_downsample_image(archipelago_noise, 10)
    show_downsample_image(archipelago_noise, 4)
