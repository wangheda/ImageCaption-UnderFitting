import json
import sys

id2captions = {}

for line in sys.stdin:
  image_id, caption, score = line.strip().split("\t")
  score = float(score)
  if image_id in id2captions:
    if id2captions[image_id][1] < score:
      id2captions[image_id] = (caption, score)
  else:
    id2captions[image_id] = (caption, score)
  
results = []
for image_id in id2captions:
  caption, score = id2captions[image_id]
  caption = caption.replace("<UNK>", "")
  result = {}
  result['image_id'] = image_id
  result['caption'] = caption
  results.append(result)

json.dump(results, sys.stdout, ensure_ascii=False, indent=4)
