#!/usr/bin/bash
echo "############## TRAINING STAGE -1 ##################"
sh train_stage1.sh
echo "############## TESTING STAGE -1 ##################"
sh test_stage1.sh
echo "############## TRAINING STAGE -2 ##################"
sh train_stage2.sh
echo "############## TESTING STAGE -2 ##################"
sh test_stage2.sh
