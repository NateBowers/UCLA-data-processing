import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 0V, Loc 1
delays = [82.3, 103.9, 98.3, 103.1, 83.9, 83.9, 135.9, 127.9, 139.1, 145.9, 131.1, 139.9, 143.5, 137.1, 147.1, 144.7, 143.5, 147.5, 159.5, 152.3, 148.7, 201.5, 187.1, 197.5, 201.5, 190.7]


def largest_within_tolerance(delays, x):
	"""Return the size and one example subset of delays where all values are within x (ns),
	including original indices from the input list.

	Args:
		delays (iterable of float): delays in ns
		x (float): tolerance in ns

	Returns:
		tuple: (count, subset_values, subset_indices)
			count (int): maximum number of delays within tolerance
			subset_values (list): sorted example subset values achieving that count
			subset_indices (list): original indices corresponding to the values
	"""
	if delays is None:
		return 0, [], []

	paired = sorted((float(d), idx) for idx, d in enumerate(delays))
	n = len(paired)
	if n == 0:
		return 0, [], []

	# find best (largest) window size using sliding window
	best_size = 0
	best_range = (0, -1)
	left = 0
	for right in range(n):
		while left <= right and paired[right][0] - paired[left][0] > x:
			left += 1
		size = right - left + 1
		if size > best_size:
			best_size = size
			best_range = (left, right)

	l, r = best_range
	subset_values = [paired[i][0] for i in range(l, r + 1)]
	subset_indices = [paired[i][1] for i in range(l, r + 1)]

	return best_size, subset_values, subset_indices


if __name__ == "__main__":

	# Sort `delays` while keeping original indices
	sorted_with_indices = sorted((val, idx) for idx, val in enumerate(delays))
	sorted_values = [v for v, _ in sorted_with_indices]
	sorted_indices = [i for _, i in sorted_with_indices]
	print(sorted_values)
	print(sorted_indices)

	# If you want to use these programmatically, they are available as
	# `sorted_values` and `sorted_indices`.