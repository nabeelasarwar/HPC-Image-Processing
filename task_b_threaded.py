# =============================================================
# Task B: Parallel Histogram using Numba CPU Threading
# Comparing Serial vs Parallel execution
# =============================================================

import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit, prange

# =============================================================
# STEP 1 - Load Image
# =============================================================
print("Loading image...")
img = Image.open("images/test_image.jpg").convert("L")
img_array = np.array(img, dtype=np.uint8)
print(f"Image loaded. Size: {img_array.shape}")
print(f"Total pixels: {img_array.size:,}")

# =============================================================
# STEP 2 - Serial Histogram (for comparison)
# This runs on 1 core only - same as before
# =============================================================
print("\n--- SERIAL HISTOGRAM ---")
start_serial = time.time()

serial_histogram, _ = np.histogram(img_array, bins=256, range=(0, 255))
serial_mean = np.mean(img_array)
serial_var = np.var(img_array)

end_serial = time.time()
serial_time = end_serial - start_serial
print(f"Serial time: {serial_time:.4f} seconds")
print(f"Mean: {serial_mean:.2f}, Variance: {serial_var:.2f}")

# =============================================================
# STEP 3 - Numba Parallel Histogram
# This is the NEW parallel version
# 
# @njit(parallel=True) tells Numba:
#   - compile this function to fast machine code
#   - enable parallel execution
#
# prange instead of range tells Numba:
#   - split this loop across all CPU cores
#   - each core handles a chunk of pixels
# =============================================================

@njit(parallel=True)
def parallel_histogram(img_flat):
    # Give each thread completely separate chunk of data
    # No sharing of memory between threads at all
    n_threads = 4
    n_pixels = len(img_flat)
    
    # Each thread gets its own private histogram row
    local_histograms = np.zeros((n_threads, 256), dtype=np.int64)
    
    # Calculate chunk size for each thread
    chunk_size = n_pixels // n_threads
    
    # Each thread processes its OWN separate chunk
    # Thread 0 → pixels 0 to chunk_size
    # Thread 1 → pixels chunk_size to 2*chunk_size
    # Thread 2 → pixels 2*chunk_size to 3*chunk_size
    # Thread 3 → pixels 3*chunk_size to end
    for t in prange(n_threads):
        start = t * chunk_size
        # Last thread takes remaining pixels
        end = n_pixels if t == n_threads - 1 else (t + 1) * chunk_size
        
        # Each thread ONLY writes to its own row
        # local_histograms[0] belongs to thread 0 only
        # local_histograms[1] belongs to thread 1 only
        # No two threads ever touch the same memory
        for i in range(start, end):
            local_histograms[t][img_flat[i]] += 1
    
    # Combine after ALL threads are done
    final_histogram = np.zeros(256, dtype=np.int64)
    for t in range(n_threads):
        for b in range(256):
            final_histogram[b] += local_histograms[t][b]
    
    return final_histogram

# =============================================================
# STEP 4 - First Run (Warm Up)
# Numba compiles your function the FIRST time it runs
# This compilation takes a few seconds but only happens once
# We do a warmup run so compilation doesnt affect our timing
# =============================================================
print("\nWarming up Numba (compiling)...")
img_flat = img_array.flatten()  # convert 2D image to 1D array
_ = parallel_histogram(img_flat)  # warmup run - dont time this
print("Compilation done!")

# =============================================================
# STEP 5 - Timed Parallel Run
# Now we time the REAL run without compilation overhead
# =============================================================
print("\n--- PARALLEL HISTOGRAM (Numba) ---")
start_parallel = time.time()

parallel_hist = parallel_histogram(img_flat)
parallel_mean = np.mean(img_flat)
parallel_var = np.var(img_flat)

end_parallel = time.time()
parallel_time = end_parallel - start_parallel
print(f"Parallel time: {parallel_time:.4f} seconds")
print(f"Mean: {parallel_mean:.2f}, Variance: {parallel_var:.2f}")

# =============================================================
# STEP 6 - Calculate Speedup
# Speedup = Serial Time / Parallel Time
# If speedup = 2.5 it means parallel is 2.5x faster
# =============================================================
print("\n--- RESULTS ---")
speedup = serial_time / parallel_time
print(f"Serial time:   {serial_time:.4f} seconds")
print(f"Parallel time: {parallel_time:.4f} seconds")
print(f"Speedup:       {speedup:.2f}x faster")
print(f"Cores used:    4")

# =============================================================
# STEP 7 - Verify Results are Correct
# Both histograms should be identical
# This proves parallelism didnt break anything
# =============================================================
if np.array_equal(serial_histogram, parallel_hist):
    print("\nVerification: PASSED - Both histograms are identical!")
else:
    print("\nVerification: FAILED - Results dont match!")

# =============================================================
# STEP 8 - Plot Comparison
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(256), serial_histogram, 
            width=1, color="blue", alpha=0.7)
axes[0].set_title(f"Serial Histogram\nTime: {serial_time:.4f}s")
axes[0].set_xlabel("Pixel Intensity")
axes[0].set_ylabel("Frequency")

axes[1].bar(range(256), parallel_hist, 
            width=1, color="green", alpha=0.7)
axes[1].set_title(f"Parallel Histogram (Numba)\nTime: {parallel_time:.4f}s")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")

plt.suptitle(f"Speedup: {speedup:.2f}x | Cores: 4", 
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("images/histogram_comparison.png")
plt.show()
print("Comparison plot saved!")