python=python3

for j in 1; do
echo "---=== XGBoost Higgs ($j threads) ===---"
$python xg.py -d  ../testdata/higgs_1000examples_test.libsvm \
              -m ../testdata/xghiggs.model \
              -t ../testdata/xghiggs_1000examples_true_predictions.txt \
              -j $j
done

for j in 1 4; do
echo "---=== lghiggs.py ($j threads) ===---"
$python lg.py -d  ../testdata/higgs_1000examples_test.libsvm \
              -m ../testdata/lghiggs.model \
              -t ../testdata/lghiggs_1000examples_true_predictions.txt \
              -j $j
done

for j in 1 4; do
echo "---=== lgmsltr.py ($j threads) ===---"
$python lg.py -d  ../testdata/msltr_1000examples_test.libsvm \
              -m ../testdata/lgmsltr.model \
              -t ../testdata/lgmsltr_1000examples_true_predictions.txt \
              -j $j
done
