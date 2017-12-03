
import sys
import copy

weights = map(int, sys.argv[1:])

for i in xrange(len(weights)):
  new_weights = copy.copy(weights)
  for w in xrange(0, 22, 2):
    new_weights[i] = w
    print "_".join(map(str, new_weights))
