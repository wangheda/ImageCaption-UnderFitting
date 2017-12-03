

dirname=$(echo $(python tools/get_voting_config.py voting.eval.config $*) | md5sum | cut -d' ' -f 1)
[ ! -d voting/$dirname ] && mkdir -p voting/$dirname

VALIDATE_REFERENCE_FILE=data/ai_challenger_caption_validation_20170910/reference.json
json_file=voting/${dirname}/vote.eval.json
result_file=voting/${dirname}/vote.eval.result
err_file=voting/${dirname}/vote.eval.err

if [ ! -f $result_file ]; then
  echo $* > voting/${dirname}/vote.eval.weights
  python tools/captions_vote.py $(python tools/get_voting_config.py voting.eval.config $*) > $json_file
  python tools/eval/run_evaluations.py -submit $json_file -ref $VALIDATE_REFERENCE_FILE > $result_file 2> $err_file
else
  echo $result_file already exists, if you want to compute again you should delete it
fi

echo $* $(tail -n 1 $result_file)
