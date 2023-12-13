#!/bin/sh
python unlearn_identified_features.py --topk 1 --alpha 0.1
python unlearn_identified_features.py --topk 2 --alpha 0.1
python unlearn_identified_features.py --topk 5 --alpha 0.1
python unlearn_identified_features.py --topk 10 --alpha 0.1