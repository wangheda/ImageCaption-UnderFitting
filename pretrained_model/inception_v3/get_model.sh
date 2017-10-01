#!/bin/bash

wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
sha1sum inception_v3_2016_08_28.tar.gz
echo "sha1sum should be: 59cd88302e8f63a7b0eaf00146d0f21df800560f"
tar xzf inception_v3_2016_08_28.tar.gz 
rm inception_v3_2016_08_28.tar.gz 
