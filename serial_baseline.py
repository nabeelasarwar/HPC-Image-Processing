# =============================================================
# Serial Baseline Script
# Task A: Sobel Edge Detection
# Task B: Intensity Histogram
# No parallelism yet - this is our starting point
# =============================================================

import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

# =============================================================
# STEP 1 - Load the image
# =============================================================
print("Loading image...")
img = Image.open("images/test_image.jpg").convert("L")  # convert to grayscale
img_array = np.array(img)
print(f"Image loaded. Size: {img_array.shape}")

# =============================================================
# STEP 2 - Task A: Sobel Edge Detection
# =============================================================
print("\nRunning Sobel Edge Detection...")
start_a = time.time()

sobel_x = sobel(img_array, axis=1)  # horizontal edges
sobel_y = sobel(img_array, axis=0)  # vertical edges
sobel_result = np.hypot(sobel_x, sobel_y)  # combine both
sobel_result = (sobel_result / sobel_result.max()) * 255  # normalize to 0-255

end_a = time.time()
print(f"Sobel completed in {end_a - start_a:.4f} seconds")

# =============================================================
# STEP 3 - Task B: Intensity Histogram
# =============================================================
print("\nComputing Intensity Histogram...")
start_b = time.time()

histogram, bin_edges = np.histogram(img_array, bins=256, range=(0, 255))
mean_intensity = np.mean(img_array)
variance_intensity = np.var(img_array)

end_b = time.time()
print(f"Histogram completed in {end_b - start_b:.4f} seconds")
print(f"Mean Intensity: {mean_intensity:.2f}")
print(f"Variance: {variance_intensity:.2f}")

# =============================================================
# STEP 4 - Save results
# =============================================================
sobel_image = Image.fromarray(sobel_result.astype(np.uint8))
sobel_image.save("images/sobel_output.jpg")
print("\nSobel output saved to images/sobel_output.jpg")

# =============================================================
# STEP 5 - Display everything
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(img_array, cmap="gray")
axes[0].set_title("Original Image (Grayscale)")
axes[0].axis("off")

axes[1].imshow(sobel_result, cmap="gray")
axes[1].set_title("Sobel Edge Detection Output")
axes[1].axis("off")

axes[2].bar(bin_edges[:-1], histogram, width=1, color="blue", alpha=0.7)
axes[2].set_title(f"Intensity Histogram\nMean={mean_intensity:.1f}, Var={variance_intensity:.1f}")
axes[2].set_xlabel("Pixel Intensity")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("images/results_plot.png")
plt.show()
print("Results plot saved to images/results_plot.png")

# =============================================================
# STEP 6 - Print total time
# =============================================================
total_time = (end_a - start_a) + (end_b - start_b)
print(f"\nTotal serial execution time: {total_time:.4f} seconds")
print("Serial baseline complete!")