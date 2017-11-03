
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

number = re.compile("[0-9]+")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    eval_file_list = []
    directory = sys.argv[1]
    for d in os.listdir(directory):
      epoch = number.findall(d)
      if d.endswith(".eval") and epoch:
        eval_file = os.path.join(directory, d, "out.eval")
        if os.path.isfile(eval_file):
          eval_file_list.append((int(epoch[0]), eval_file))
    eval_file_list.sort()
    print("| epoch | Bleu_4 | CIDEr | METEOR | ROUGE_L |")
    print("|:------|-------:|------:|-------:|--------:|")
    for epoch, eval_file in eval_file_list:
      try:
        obj = score(eval_file)
        B4 = obj["Bleu_4"]
        C = obj["CIDEr"]
        M = obj["METEOR"]
        R = obj["ROUGE_L"]
        print("| %d | %.4f | %.4f | %.4f | %.4f |" % (epoch, B4, C, M, R))
      except Exception as e:
        #print(e)
        pass



    
          

        
  else:
    sys.exit("Usage: python %s [model_dir]" % sys.argv[0])
