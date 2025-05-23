# call from top-level folder, i.e., <sh ./bash/create_datasets.sh>

cmd="python vpc_inference.py \
--ckpt_model ./ckpt/model_base.pth \
--ckpt_voc ./vocoder/voc_ckpt.pth \
--ckpt_f0_vqvae ./f0_vqvae/f0_vqvae.pth \
--time_step 6 \
--theta 0"

$cmd --method VoiceVMF --epsilon 1
$cmd --method VoiceVMF --epsilon 10
#$cmd --method VoiceVMF --epsilon 50
$cmd --method VoiceVMF --epsilon 100
#$cmd --method VoiceVMF --epsilon 200

$cmd --method IdentityDP --epsilon 1
$cmd --method IdentityDP --epsilon 10
#$cmd --method IdentityDP --epsilon 50
$cmd --method IdentityDP --epsilon 100
#$cmd --method IdentityDP --epsilon 200

$cmd --method pitchshift --semitones 5
$cmd --method pitchshift --semitones -5
