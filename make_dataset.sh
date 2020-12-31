cd emotion_detection/data
wget "https://www.floydhub.com/api/v1/resources/qM4BHN3pNjkfkvYjMMtjU4/ISEAR.csv?content=true&rename=isearcsv"
file_name=$(ls | grep ISEAR)
mv $file_name ISEAR.csv