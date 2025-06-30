# Workaround for opencompass dataset downloading bug
# https://github.com/open-compass/opencompass/issues/2035

cd data

# https://github.com/open-compass/opencompass/blob/0964799e6c4016ec24cca369ab35ecb597ba2c62/opencompass/utils/datasets_info.py#L759
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
unzip aime.zip
rm aime.zip
