
import sys
import json

if __name__ == "__main__":
  captions = {}
  for filename in sys.argv[1:]:
    with open(filename) as F:
      json_list = json.loads(F.read())
    for item in json_list:
      image_id = item['image_id']
      caption = item['caption']
      if image_id in captions:
        captions[image_id].append(caption)
      else:
        captions[image_id] = [caption,]
      
  final_json_list = []
  for image_id, caps in captions.items():
    caption_count = dict([(c,0) for c in set(caps)])
    for c in caps:
      caption_count[c] += 1
    final_caption = caps[0]
    final_count = caption_count[final_caption]
    for c in set(caps):
      if caption_count[c] > final_count:
        final_count = caption_count[c]
        final_caption = c
    result = {}
    result['image_id'] = image_id
    result['caption'] = final_caption
    #print >> sys.stderr, image_id, final_caption, final_count
    final_json_list.append(result)

  sys.stdout.write(json.dumps(final_json_list, ensure_ascii=False, indent=4).encode("utf8"))

