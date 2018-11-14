python=python3

for j in 1; do
echo "---=== XGBoost Higgs ($j threads) ===---"
$python xg.py -d  ../testdata/higgs_1000examples_test.libsvm \
              -m ../testdata/xghiggs.model \
              -t ../testdata/xghiggs_1000examples_true_predictions.txt \
              -j $j
done

for j in 1 4; do
echo "---=== LightGBM Higgs ($j threads) ===---"
$python lg.py -d  ../testdata/higgs_1000examples_test.libsvm \
              -m ../testdata/lghiggs.model \
              -t ../testdata/lghiggs_1000examples_true_predictions.txt \
              -j $j
done

for j in 1 4; do
echo "---=== LightGBM MS LTR ($j threads) ===---"
$python lg.py -d  ../testdata/msltr_1000examples_test.libsvm \
              -m ../testdata/lgmsltr.model \
              -t ../testdata/lgmsltr_1000examples_true_predictions.txt \
              -j $j
done

# do not forget run before to generate data:
# > cd testdata
# > $python lg_kddcup99.py bench
for j in 1 4; do
echo "---=== LightGBM KDD Cup 99 ($j threads) ===---"
$python lg.py -d  ../testdata/kddcup99_test_for_bench.tsv \
              -m ../testdata/lg_kddcup99_for_bench.model \
              -t ../testdata/lg_kddcup99_true_predictions_for_bench.txt \
              -j $j
done
