# encoding: utf-8
# build reference file for evaluating attributes predictor
# Just segment each caption of a image
# __author__ = 'Miao'
# python2.7
# python run_attributes_evaluations.py --submit=your_result_json_file --ref=attributes_reference_json_file

import sys
import argparse
import json
import jieba


reload(sys)
sys.setdefaultencoding('utf8')


def main():
    """The evaluator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation",
                        help=' JSON containing annotation.')
    parser.add_argument("output",
                        help=' output JSON containing references.')
    args = parser.parse_args()

    input = open(args.annotation)
    all_samples = json.load(input)
    input.close()

    for sample in all_samples:
        captions = sample['caption']
        seg_captions = []
        for c in captions:
            c = c.strip().strip(u"。").replace('\n', '')
            seg_list = jieba.cut(c, cut_all=False)
            seg_captions.append(" ".join(seg_list))
        sample['caption'] = seg_captions

    output = open(args.output, 'w')
    json.dump(all_samples, output, ensure_ascii=False, indent=4)
    output.close()


if __name__ == "__main__":
    main()
