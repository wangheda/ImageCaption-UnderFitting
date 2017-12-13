

dirname=$(echo $(python tools/get_voting_config.py voting.test2.config $*) | md5sum | cut -d' ' -f 1)
[ ! -d voting-test2/$dirname ] && mkdir -p voting-test2/$dirname

VALIDATE_REFERENCE_FILE=data/ai_challenger_caption_validation_20170910/reference.json
json_file=voting-test2/${dirname}/vote.test2.json

if [ ! -f $json_file ]; then
  for file in $(python tools/get_voting_config.py voting.test2.config $*); do
    [ ! -f $file ] && echo $file does not exist && exit 1
  done

  echo $* > voting-test2/${dirname}/vote.test2.weights
  python tools/captions_vote.py $(python tools/get_voting_config.py voting.test2.config $*) > $json_file
else
  echo $json_file already exists, if you want to compute again you should delete it
fi
