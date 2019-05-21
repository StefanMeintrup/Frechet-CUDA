#!/bin/bash

# download data

echo "Downloading data...."
wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip'

#extract data

echo "Extracting data..."
unzip data.zip

#deleting zip

echo "Deleting zip...."
rm data.zip

