EncoderArray=("use" "attn_dssm_4" "dssm")
for i in ${EncoderArray[*]};
do
    pbk_train -e $i -t 0.1 -ep 20 -mt 32 --multi_label --parallel -bs 128 -lr 0.002 -dr 0.4 -bn
done