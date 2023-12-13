#!/bin/sh
python unlearn_identified_features.py --topk 5 --alpha 0.1
python unlearn_identified_features.py --topk 5 --alpha 0.2
python unlearn_identified_features.py --topk 5 --alpha 0.5
python unlearn_identified_features.py --topk 5 --alpha 1.0