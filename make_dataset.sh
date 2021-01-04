cd emotion_detection/data
wget -O ISEAR.csv "https://www.floydhub.com/api/v1/resources/qM4BHN3pNjkfkvYjMMtjU4/ISEAR.csv?content=true&rename=isearcsv"
kaggle datasets download -d mirosval/personal-cars-classifieds
unzip -o personal-cars-classifieds
rm -rf personal-cars-classifieds.zip