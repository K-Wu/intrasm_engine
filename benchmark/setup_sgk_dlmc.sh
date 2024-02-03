# Download  https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz and extract it to the ../data/sgk_dlmc folder
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz

# Find the script path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
mkdir -p ${DIR}/../data/sgk_dlmc
tar -xvf dlmc.tar.gz -C ${DIR}/../data/sgk_dlmc
rm dlmc.tar.gz