# call from top-level folder, i.e., <sh ./bash/create_datasets.sh>

cmd="python inference_batch.py \
--audio_folder D:/vox1_test_wav \
--ckpt_model ./ckpt/model_base.pth \
--ckpt_voc ./vocoder/voc_ckpt.pth \
--ckpt_f0_vqvae ./f0_vqvae/f0_vqvae.pth \
--time_step 6 \
--theta 0"

# $cmd --method VoiceVMF --epsilon 1
# $cmd --method VoiceVMF --epsilon 10
# $cmd --method VoiceVMF --epsilon 50
# $cmd --method VoiceVMF --epsilon 100
# $cmd --method VoiceVMF --epsilon 200

# $cmd --method IdentityDP --epsilon 1
# $cmd --method IdentityDP --epsilon 10
# $cmd --method IdentityDP --epsilon 50
# $cmd --method IdentityDP --epsilon 100
# $cmd --method IdentityDP --epsilon 200

cmd2="python inference_pitchshift.py \
--audio_folder D:\vox1_test_wav"

$cmd2 --semitones 1
$cmd2 --semitones 3
$cmd2 --semitones 5
$cmd2 --semitones -1
$cmd2 --semitones -3
$cmd2 --semitones -5