# call from top-level folder, i.e., <sh ./bash/create_datasets.sh>

cmd="python ./whisper/transcript.py \
--model turbo"

for dir in /blue/ejain/datasets/*/
do
    dir=${dir%*/}      # remove the trailing "/"
    echo "$dir"    # print everything after the final "/"
    $cmd --audio_folder $dir
done

