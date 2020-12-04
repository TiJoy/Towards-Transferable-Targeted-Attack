#!/bin/bash
python attacker_Po_cos_incep_v3.py --output_dir=result-incep_v3_TI_po/ --loss=po --logname=incep_v3_po.txt --cuda=2
python attacker_Po_cos_incep_v4.py --output_dir=result-incep_v4_TI_po/ --loss=po --logname=incep_v4_po.txt --cuda=2
python attacker_Po_cos_incep_res_v2.py --output_dir=result-incep_res_v2_TI_po/ --loss=po --logname=incep_res_v2_po.txt --cuda=2
python attacker_Po_cos_res_v2_50.py --output_dir=result-res_v2_50_TI_po/ --loss=po --logname=res_v2_50_po.txt --cuda=2
python attacker_Po_cos_res_v2_101.py --output_dir=result-res_v2_101_TI_po/ --loss=po --logname=res_v2_101_po.txt --cuda=2
python attacker_Po_cos_res_v2_152.py --output_dir=result-res_v2_152_TI_po/ --loss=po --logname=res_v2_152_po.txt --cuda=2
