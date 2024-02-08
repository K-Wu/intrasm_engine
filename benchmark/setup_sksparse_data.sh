wget https://github.com/scikit-sparse/scikit-sparse/raw/master/tests/test_data/illc1033.mtx.gz
wget https://github.com/scikit-sparse/scikit-sparse/raw/master/tests/test_data/illc1033_rhs1.mtx.gz
wget https://github.com/scikit-sparse/scikit-sparse/raw/master/tests/test_data/illc1850.mtx.gz
wget https://github.com/scikit-sparse/scikit-sparse/raw/master/tests/test_data/illc1850_rhs1.mtx.gz
wget https://github.com/scikit-sparse/scikit-sparse/raw/master/tests/test_data/well1033.mtx.gz
wget https://github.com/scikit-sparse/scikit-sparse/raw/master/tests/test_data/well1033_rhs1.mtx.gz
wget https://github.com/scikit-sparse/scikit-sparse/raw/master/tests/test_data/well1850.mtx.gz
wget https://github.com/scikit-sparse/scikit-sparse/raw/master/tests/test_data/well1850_rhs1.mtx.gz

# Find the script path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
mkdir -p ${DIR}/../data/sksparse
mv *.mtx.gz ${DIR}/../data/sksparse

