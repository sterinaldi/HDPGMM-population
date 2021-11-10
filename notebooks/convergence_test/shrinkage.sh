#!/bin/bash

for N in 100 500 1000 5000 10000
do
    echo $N
    python single_event.py --optfile notebooks/convergence_test/options_shrinkage.ini --n_samps_dsp $N
    mv notebooks/convergence_test/shrinkage/reconstructed_events/events/gaussian.pdf notebooks/convergence_test/shrinkage/reconstructed_events/events/gaussian_$N.pdf
    mv notebooks/convergence_test/shrinkage/reconstructed_events/events/log_gaussian.pdf notebooks/convergence_test/shrinkage/reconstructed_events/events/log_gaussian_$N.pdf
done
