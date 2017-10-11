#!/bin/python

import sys
import re

if __name__ == "__main__":
  interval = int(sys.argv[1])

  regex_num = re.compile("[0-9]+")
  points = []
  for line in sys.stdin:
    base_name = line.strip().split("/")[-1]
    if base_name.startswith("model.ckpt"):
      ckpt = regex_num.findall(base_name)
      if ckpt:
        ckpt = int(ckpt[0])
        points.append(ckpt)

  points.sort()

  intervals = xrange(0, max(points) + 1 + interval, interval)
  for i in xrange(len(intervals)-1):
    start = intervals[i]
    end = intervals[i+1]
    point_set = [p for p in points if start <= p < end]
    if point_set:
      print min(point_set)




