
import re
import os
import sys
import ast
import json

def score(filename):
  with open(filename) as F:
    last_line = None
    for line in F:
      last_line = line.strip()
  obj = ast.literal_eval(last_line)
  return obj

number = re.compile("^[0-9a-f]+$")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    eval_file_list = []
    directory = sys.argv[1]
    for d in os.listdir(directory):
      if number.match(d):
        eval_file = os.path.join(directory, d, "vote.eval.result")
        if os.path.isfile(eval_file):
          eval_file_list.append((d, eval_file))
    eval_file_list.sort()
    C_best = 0
    d_best = ""
    obj_best = {}
    for d, eval_file in eval_file_list:
      try:
        obj = score(eval_file)
        B4 = obj["Bleu_4"]
        C = obj["CIDEr"]
        M = obj["METEOR"]
        R = obj["ROUGE_L"]
        if C > C_best:
          C_best = C
          d_best = d
          obj_best = obj
      except Exception as e:
        #print(e)
        pass
    print "Cider:", C_best
    print "Path:", os.path.join(directory, d_best)
    print "Weights:", open(os.path.join(directory, d_best, "vote.eval.weights")).read().strip()
    print "Results:", obj_best
  else:
    sys.exit("Usage: python %s [model_dir]" % sys.argv[0])


