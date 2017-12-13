
import sys

config_file = sys.argv[1]
weights = map(int, sys.argv[2:])
directories = []

with open(config_file) as F:
  for line in F:
    d = line.strip()
    directories.append(d)

assert len(directories) == len(weights), "%d != %d" % (len(directories), len(weights))

for d, w in zip(directories, weights):
  for i in xrange(1, w+1):
    print "%s/run-%d.json" % (d, i)
    
