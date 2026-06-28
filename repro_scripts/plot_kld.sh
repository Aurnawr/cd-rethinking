python ./inference/kld_plot.py \
    --vcd  ./repro_outputs/kld_experiment/visit-bench/vcd_kld.json  \
    --icd  ./repro_outputs/kld_experiment/visit-bench/icd_kld.json  \
    --sid  ./repro_outputs/kld_experiment/visit-bench/sid_kld.json  \
    --max-tokens 128 \
    --smooth 10 \
    --out  ./repro_outputs/kld_experiment/visit-bench/kld_plot.png