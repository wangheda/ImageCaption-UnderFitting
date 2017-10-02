
import sys
import json

if __name__ == "__main__":
  all_json_list = []
  for filename in sys.argv[1:]:
    with open(filename) as F:
      json_list = json.loads(F.read())
    all_json_list.extend(json_list)
  sys.stdout.write(json.dumps(all_json_list, ensure_ascii=False, indent=4).encode("utf8"))

