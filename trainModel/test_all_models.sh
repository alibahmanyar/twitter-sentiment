#! /bin/bash
mkdir tests
for i in {0..6}
do
    mkdir tests/${i}
    cat test_model.py | sed 's/MODEL_NO\s=\s./MODEL_NO = '${i}'/g' > test_model_tmp.py
    ./test_model_tmp.py > tests/${i}/out.log 2>&1
done