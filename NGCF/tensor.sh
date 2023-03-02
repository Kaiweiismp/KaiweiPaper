if [ -z "$1" ]
  then
    echo "Please input the dir name of tensorboard which want to show"
  else
    if [ -z "$2" ]
      then
        summary_path="$1"
        echo "${summary_path}"
        tensorboard --logdir=$summary_path --host=140.116.82.209
      else
        summary_path="$1"
        echo "${summary_path}"
        python3 -m tensorboard.main --logdir=$summary_path --host=140.116.82.209 --port=$2
    fi
fi