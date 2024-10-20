import numpy as np
print(np.__version__)

rng = np.random.default_rng(0)

data = rng.normal(size=(2, 101))

data[:, 2] = np.nan

test = np.ma.masked_array(data)

std = np.ma.std(test, axis=1)
median = np.ma.median(test, axis=1)
mean = np.ma.mean(test, axis=1)

print("std:")
print(repr(std))
print("median:")
print(repr(median))
print("mean:")
print(repr(mean))

std = np.std(data, axis=1)
median = np.median(data, axis=1)
mean = np.mean(data, axis=1)

print("std:")
print(repr(std))
print("median:")
print(repr(median))
print("mean:")
print(repr(mean))