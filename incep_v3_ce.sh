#!/bin/bash
python attacker_Po_cos_incep_v3.py --output_dir=result-incep_v3_TI_ce/ --loss=ce --logname=incep_v3_ce.txt --cuda=1
python attacker_Po_cos_incep_v4.py --output_dir=result-incep_v4_TI_ce/ --loss=ce --logname=incep_v4_ce.txt --cuda=1
python attacker_Po_cos_incep_res_v2.py --output_dir=result-incep_res_v2_TI_ce/ --loss=ce --logname=incep_res_v2_ce.txt --cuda=1
python attacker_Po_cos_res_v2_50.py --output_dir=result-res_v2_50_TI_ce/ --loss=ce --logname=res_v2_50_ce.txt --cuda=1
python attacker_Po_cos_res_v2_101.py --output_dir=result-res_v2_101_TI_ce/ --loss=ce --logname=res_v2_101_ce.txt --cuda=1
python attacker_Po_cos_res_v2_152.py --output_dir=result-res_v2_152_TI_ce/ --loss=ce --logname=res_v2_152_ce.txt --cuda=1
