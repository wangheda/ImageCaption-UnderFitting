
for i in $(python tools/local_search.py $*); do 
  echo "bash voting-eval.sh $(echo $i | sed 's/_/ /g')"
done 
