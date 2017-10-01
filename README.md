Team code repository for image caption task
-------

# Setting up the data path
  
Use link to point to the data directory

    ln -s [the-path-to-data] data
    
Now the relative path of training, validation, testing data (with annotations) should be like:

    data/ai_challenger_caption_train_20170902/caption_train_images_20170902/*.jpg
    data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json

    data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/*.jpg
    data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json

    data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/*.jpg

# Generate the tfrecord files

If there are no `TFRecord_data` in directory `data`, it is likely the the tfrecord files have not been 
generated yet, run the script (after setting up the data path):

    bash scripts/build_tfrecords.sh

to create the tfrecord files. It may take 20-30 minutes to create all the files.

# Training 

First, you need to get the inception v3 network checkpoint, goto directory 
`pretrained_model/inception_v3`, run:

    bash get_model.sh

It will automatically download the checkpoint.

Then, you can run `bash train.sh` for baseline.

You can also create another training script with a different configuration.

# Validate

Pls rewrite eval.py and put any supportive scripts needed in `eval_utils`, do not change `im2txt_model.py` unless necessary.

Write an `eval.sh` similar to `train.sh`.

# Inference

Pls rewrite inference.py and put any upportive scripts needed in `inference_utils`, do not change `im2txt_model.py` unless necessary.

Write an `inference.sh` similar to `train.sh`.

# Important notice

1. Commit as soon as your branch is merged with `origin/master` and tested, beware of silent merge conflict.
2. Commit from where you edit. DO NOT edit on windows, transmit to linux, and commit on linux (or vice versa) as it will cause different line ending.

