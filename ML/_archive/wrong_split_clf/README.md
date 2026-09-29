# Classifier models trained on the WRONG split — do not cite

`clf_train.py` hardcoded route seeds 1-17 (fit) / 18-21 (stop) and was missed
when the protocol moved to 1-19 / 20-21. Every classifier here therefore
trained on 569 routes instead of 639 and stopped on a different slice from the
trees and MLP. No test leakage (test is 22-25 throughout), but the classifier
was at a data disadvantage in every comparison it appeared in.

Fixed in clf_train.py (it now imports FIT_SEEDS / STOP_SEEDS / TEST_SEEDS from
dataset.py). All classifier rows are retrained under the correct protocol.
