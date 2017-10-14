#! /usr/bin/python
# -*- coding: utf-8 -*- 

"""
Data augmentation by fliping the image
"""

import os
from PIL import Image
import json



caption_sample = u"一个左右手拿着东西，左手插兜的女人走在房屋前的右侧道路上"

def find_all(string, query):
    # return all positions
    query_len = len(query)
    positions = []
    beg = 0
    pos = string.find(query, beg)
    while pos != -1:
        positions.append(pos)
        beg = pos + query_len
        pos = string.find(query, beg)
    return positions




def flip_caption(caption):
    lr_pos = find_all(caption, u"左右")
    noflip_pos = []
    for pos in lr_pos:
        noflip_pos.append(pos)
        noflip_pos.append(pos + 1)
    l_pos = find_all(caption, u"左")
    l_pos = [pos for pos in l_pos if pos not in noflip_pos]

    r_pos = find_all(caption, u"右")
    r_pos = [pos for pos in r_pos if pos not in noflip_pos]

    if not l_pos and not r_pos:
        return caption

    new_caption = ""
    for i,c in enumerate(caption):
        if i in l_pos:
            new_caption += u"右"
        elif i in r_pos:
            new_caption += u"左"
        else:
            new_caption += c
    return new_caption

#print caption_sample
#print flip_caption(caption_sample)

# the path here should be set correctly
data_path = "../data/ai_challenger_caption_train_20170902"
annotation_file = os.path.join(data_path, "caption_train_annotations_20170902.json")
image_path = os.path.join(data_path, "caption_train_images_20170902")

output_annotation_file =  "../data/aug_train_annotations.json"

input = open(annotation_file, 'r')
train_samples = json.load(input)
input.close()



for i,sample in enumerate(train_samples):
    if i % 1000 == 0:
        print i
    captions = sample['caption']
    image_id = sample['image_id']
    sample['flip_caption'] = []
    for caption in captions:
        sample['flip_caption'].append(flip_caption(caption))

output = open(output_annotation_file, 'w')
json.dump(train_samples, output, indent=4)
output.close()


