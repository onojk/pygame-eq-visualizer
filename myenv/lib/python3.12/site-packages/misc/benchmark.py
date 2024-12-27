import time

import numpy as np
import scipy.special

import audmath


np.random.seed(0)


print("\nExecution time for inverse normal distribution\n")
print("Samples   scipy audmath")
for samples in [
    10000,
    100000,
    1000000,
    10000000,
]:
    y = np.random.random_sample((samples,))

    # Benchmark audmath
    start = time.process_time()
    x = audmath.inverse_normal_distribution(y)
    end = time.process_time()
    time_audmath = end - start

    # Benchmark scipy
    start = time.process_time()
    x = scipy.special.ndtri(y)
    end = time.process_time()
    time_scipy = end - start

    print(f"{samples: 9.0f} {time_scipy:.2f}s {time_audmath:.2f}s")
