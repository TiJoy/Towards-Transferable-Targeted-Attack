#!/bin/bash
python attacker_Po_cos_incep_v3.py --output_dir=/data/ltj/Pycharm_Projects/Po-Attack/result-incep_v3_TI_trip_po_time/ --loss=trip_po --logname=incep_v3_trip_po.txt --cuda=0
python attacker_Po_cos_incep_v4.py --output_dir=/data/ltj/Pycharm_Projects/Po-Attack/result-incep_v4_TI_trip_po_time/ --loss=trip_po --logname=incep_v4_trip_po.txt --cuda=0
python attacker_Po_cos_incep_res_v2.py --output_dir=/data/ltj/Pycharm_Projects/Po-Attack/result-incep_res_v2_TI_trip_po_time/ --loss=trip_po --logname=incep_res_v2_trip_po.txt --cuda=0
python attacker_Po_cos_res_v2_50.py --output_dir=/data/ltj/Pycharm_Projects/Po-Attack/result-res_v2_50_TI_trip_po_time/ --loss=trip_po --logname=res_v2_50_trip_po.txt --cuda=0
python attacker_Po_cos_res_v2_101.py --output_dir=/data/ltj/Pycharm_Projects/Po-Attack/result-res_v2_101_TI_trip_po_time/ --loss=trip_po --logname=res_v2_101_trip_po.txt --cuda=0
python attacker_Po_cos_res_v2_152.py --output_dir=/data/ltj/Pycharm_Projects/Po-Attack/result-res_v2_152_TI_trip_po_time/ --loss=trip_po --logname=res_v2_152_trip_po.txt --cuda=0