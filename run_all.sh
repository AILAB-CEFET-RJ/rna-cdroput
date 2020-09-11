#!/bin/bash

nohup python main.py -noes -e 3000 -runs 10 -lr 0.001 -f 15 -cut -subs 120000 -dp ErrorBasedDropoutIR -sc StandardScaler -dataset sdss -hl1 790 -hl2 158 -noes > out_test_ss_sdss.txt;

nohup python main.py -noes -e 3000 -runs 10 -lr 0.001 -f 15 -coin_val B -dp ErrorBasedDropoutIR -sc StandardScaler -dataset teddy -hl1 790 -hl2 158 -noes > out_test_ss_teddy_B.txt;
nohup python main.py -noes -e 3000 -runs 10 -lr 0.001 -f 15 -coin_val C -mo -dp ErrorBasedDropoutIR -sc StandardScaler -dataset teddy -hl1 790 -hl2 158 -noes > out_test_ss_teddy_C.txt;
nohup python main.py -noes -e 3000 -runs 10 -lr 0.001 -f 15 -coin_val D -mo -dp ErrorBasedDropoutIR -sc StandardScaler -dataset teddy -hl1 790 -hl2 158 -noes > out_test_ss_teddy_D.txt;

nohup python main.py -noes -e 3000 -runs 10 -lr 0.001 -f 15 -coin_val B -dp ErrorBasedDropoutIR -sc StandardScaler -dataset happy -hl1 790 -hl2 158 -noes > out_test_ss_happy_B.txt;
nohup python main.py -noes -e 3000 -runs 10 -lr 0.001 -f 15 -coin_val C -mo -dp ErrorBasedDropoutIR -sc StandardScaler -dataset happy -hl1 790 -hl2 158 -noes > out_test_ss_happy_C.txt;
nohup python main.py -noes -e 3000 -runs 10 -lr 0.001 -f 15 -coin_val D -mo -dp ErrorBasedDropoutIR -sc StandardScaler -dataset happy -hl1 790 -hl2 158 -noes > out_test_ss_happy_D.txt;

