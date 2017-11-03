# encoding: utf-8
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation utility for image Chinese captioning task."""
# __author__ = 'Miao'
# python2.7
# python run_attributes_evaluations.py --submit=your_result_json_file --ref=attributes_reference_json_file

import sys
import argparse
import json
import codecs
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')




def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def compute_map(json_predictions_file, reference_file, attributes_file, k_list = [5,10,15,20]):
    """Compute map"""
    input = open(json_predictions_file)
    predictions = json.load(input)
    input.close()

    input = open(reference_file)
    reference = json.load(input)
    reference_dict = {}
    for sample in reference:
        image_id = sample['image_id']
        captions = sample['caption']
        reference_dict[image_id] = captions
    input.close()

    input = codecs.open(attributes_file, 'r', 'utf-8')
    attributes = [line.split(" ")[0] for line in input.readlines()]
    attributes = set(attributes)
    input.close()

    map_scores = {}
    for k in k_list:
        map_scores[k] = []

    for pred in predictions:
        predicted_attributes = pred['attributes']
        predicted_attributes = [attr for attr in predicted_attributes.split(" ") if attr]
        refs = reference_dict[pred['image_id'] + ".jpg"]
        for ref in refs:
            ref = ref.split(" ")
            ref = [word for word in ref if word in attributes]
            ref_attributes = list(set(ref))
            for k in k_list:
                map_scores[k].append(apk(ref_attributes, predicted_attributes, k))

    for k in k_list:
        map_scores[k] = np.mean(map_scores[k])
    return map_scores


def main():
    """The evaluator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-submit", "--submit", type=str, required=True,
                        help=' JSON containing submit sentences.')
    parser.add_argument("-ref", "--ref", type=str,
                        help=' JSON references.')
    parser.add_argument("-attr", "--attr", type=str,
                        help='attributes file.')
    args = parser.parse_args()

    json_predictions_file = args.submit
    reference_file = args.ref
    print compute_map(json_predictions_file, reference_file, args.attr)


if __name__ == "__main__":
    main()
