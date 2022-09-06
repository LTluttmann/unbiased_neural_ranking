There might be some issues depending on the computing instance when running the scripts defined here from the command line. To resolve some scipy import error, the path to the anaconda environment has to be added to the LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/
```

This has been added to the `install_required_on_gpu_nb.sh` script, in order to fix the issue on AWS EC-2 instances (with GPU).

- To run the pbk classification use (note, flags might be subject to change): 
```bash
pbk_train -e attn_dssm_4 -t 0.1 -ep 10 -mt 32 --multi_label --parallel -bs 128 -lr 0.002 -dr 0.4 -bn
```

- To run the encoder training use: TODO

```bash
encoder_train -e use -bs 200 -l bpr -nn 10 -mt 32 -s1 20000 -s2 80000
```

- To run the ranker training use: TODO