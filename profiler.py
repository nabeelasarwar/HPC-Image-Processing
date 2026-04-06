import cProfile
import pstats
import numpy as np
from PIL import Image
from scipy.ndimage import sobel

# Load image
img = Image.open("images/test_image.jpg").convert("L")
img_array = np.array(img)

def run_sobel():
    sobel_x = sobel(img_array, axis=1)
    sobel_y = sobel(img_array, axis=0)
    result = np.hypot(sobel_x, sobel_y)
    return result

def run_histogram():
    histogram, _ = np.histogram(img_array, bins=256, range=(0, 255))
    mean = np.mean(img_array)
    var = np.var(img_array)
    return histogram, mean, var

# Profile both functions
profiler = cProfile.Profile()
profiler.enable()

run_sobel()
run_histogram()

profiler.disable()

# Save clean results
with open("profiling_clean.txt", "w") as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

print("Profiling complete! Check profiling_clean.txt")