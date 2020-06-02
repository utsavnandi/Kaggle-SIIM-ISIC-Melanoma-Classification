pip uninstall kaggle -y
pip install kaggle==1.5.6 -q
mkdir ~/.kaggle/
cp ./kaggle.json  ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download tunguz/siimisic-melanoma-resized-images -f x_train_32.npy
kaggle datasets download tunguz/siimisic-melanoma-resized-images -f x_test_32.npy
unzip ./680899%2F1200702%2Fcompressed%2Fx_train_32.npy.zip -d ./data/
unzip ./680899%2F1200702%2Fcompressed%2Fx_test_32.npy.zip -d ./data/
rm ./680899%2F1200702%2Fcompressed%2Fx_train_32.npy.zip
rm ./680899%2F1200702%2Fcompressed%2Fx_test_32.npy.zip
kaggle competitions download siim-isic-melanoma-classification -f sample_submission.csv
kaggle competitions download siim-isic-melanoma-classification -f test.csv
kaggle competitions download siim-isic-melanoma-classification -f train.csv
unzip train.csv -d ./data/
mv ./test.csv ./data/
mv ./sample_submission.csv ./data/
rm train.csv.zip
mkdir ./logs/