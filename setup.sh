pip uninstall torch
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip uninstall kaggle -y
pip install kaggle==1.5.6 -q
pip install -U git+https://github.com/albu/albumentations -q
pip install -U git+https://github.com/rwightman/pytorch-image-models -q
pip install neptune-client -q
mkdir ~/.kaggle/
cp /home/utsav_nandi/kaggle.json  ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d shonenkov/melanoma-merged-external-data-512x512-jpeg
unzip melanoma-merged-external-data-512x512-jpeg.zip -d /home/utsav_nandi/data/
rm /home/utsav_nandi/melanoma-merged-external-data-512x512-jpeg.zip
kaggle competitions download siim-isic-melanoma-classification -f sample_submission.csv
kaggle competitions download siim-isic-melanoma-classification -f test.csv
kaggle competitions download siim-isic-melanoma-classification -f train.csv
unzip /home/utsav_nandi/train.csv -d /home/utsav_nandi/data/
mv /home/utsav_nandi/test.csv /home/utsav_nandi/data/
mv /home/utsav_nandi/sample_submission.csv /home/utsav_nandi/data/
rm /home/utsav_nandi/train.csv.zip
mkdir /home/utsav_nandi/logs/