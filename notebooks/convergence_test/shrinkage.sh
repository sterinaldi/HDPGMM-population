#!/bin/bash

for N in 100 177 316 500 562 1000 1778 3162 5000 5623 10000
do
    echo $N
    python single_event.py --optfile notebooks/convergence_test/options_shrinkage.ini --n_samps_dsp $N
    mv notebooks/convergence_test/shrinkage/reconstructed_events/events/gaussian.pdf notebooks/convergence_test/shrinkage/reconstructed_events/events/gaussian_$N.pdf
    mv notebooks/convergence_test/shrinkage/reconstructed_events/events/log_gaussian.pdf notebooks/convergence_test/shrinkage/reconstructed_events/events/log_gaussian_$N.pdf
    mv notebooks/convergence_test/shrinkage/reconstructed_events/rec_prob/log_rec_prob_gaussian.txt notebooks/convergence_test/shrinkage/reconstructed_events/rec_prob/log_rec_prob_gaussian_$N.txt
done
