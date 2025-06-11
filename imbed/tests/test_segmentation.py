"""

The following are tests that demo the workings of fixed_step_chunker


>>> from imbed.segmentation_util import fixed_step_chunker

>>> # testing chk_step < chk_size with return_tail=TRUE, stop and start_at PRESENT
>>> # and stop_at SMALLER than the largest index of it
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=2, stop_at=5, return_tail=True)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[3, 4, 5], [4, 5], [5]]

# testing chk_step < chk_size with return_tail=FALSE, stop and start_at PRESENT
# and stop_at SMALLER than the largest index of it
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=2, stop_at=5, return_tail=False)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[3, 4, 5]]

# testing chk_step < chk_size with return_tail=TRUE, stop and start_at PRESENT
# and stop_at LARGER than the largest index of it
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=1, stop_at=20, return_tail=True)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16], [15, 16], [16]]

# testing chk_step < chk_size with return_tail=FALSE, stop and start_at PRESENT
# and stop_at LARGER than the largest index of it
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=1, stop_at=20, return_tail=False)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16]]

# testing chk_step = chk_size with return_tail=TRUE, stop and start_at PRESENT
# and stop_at SMALLER than the largest index of it
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=3, start_at=1, stop_at=7, return_tail=True)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[2, 3, 4], [5, 6, 7]]

# testing chk_size > len(it) with return_tail=False, no stop_at or start_at
>>> f = lambda it: fixed_step_chunker(it, chk_size=30, chk_step=3, start_at=None, stop_at=None, return_tail=False)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[]

# testing chk_size > len(it) with return_tail=True, no stop_at or start_at
>>> f = lambda it: fixed_step_chunker(it, chk_size=30, chk_step=3, start_at=None, stop_at=None, return_tail=True)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [10, 11, 12, 13, 14, 15, 16], [13, 14, 15, 16], [16]]

# testing chk_step > chk_size with return_tail=TRUE, stop and start_at PRESENT
# and stop_at SMALLER than the largest index of it
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=1, stop_at=7, return_tail=True)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[2, 3, 4], [6, 7]]

# testing chk_step > chk_size with return_tail=FALSE, stop and start_at PRESENT
# and stop_at SMALLER than the largest index of it
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=1, stop_at=7, return_tail=False)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[2, 3, 4]]

# testing chk_step > chk_size with return_tail=FALSE, stop and start_at NOT PRESENT
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=False)
>>> it = range(1, 17, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]]

# testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=True)
>>> it = range(1, 19, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18]]

# testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
# with negative values in the iterator
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=True)
>>> it = range(-10, 19, 1)
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[[-10, -9, -8], [-6, -5, -4], [-2, -1, 0], [2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16], [18]]

# testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
# with items of various types in the iterator
>>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=2, start_at=None, stop_at=None, return_tail=True)
>>> it = ['a', 3, -10, 9.2, str, [1,2,3], set([10,20])]
>>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
>>> assert A == B  # it and iter(it) should give the same thing!
>>> A  # and that thing is:
[['a', 3, -10], [-10, 9.2, <class 'str'>], [<class 'str'>, [1, 2, 3], {10, 20}], [{10, 20}]]
"""
