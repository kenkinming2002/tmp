# Usage
 - Download an unpack data from
   https://drive.google.com/drive/folders/1R2BdIpQxDqxo6pIbr0ajlX44eRbvqKcz
   so that there is a directory named data.
 - pip install -r requirements.txt
 - ./src/preprocess.py
 - ./src/train.py or ./src/train\_torch.py

# Note
As of now, the model does not train on the dataset.
Since the model does not train both using the self implemented version and with
pytorch, my guess would be that something is wrong during the data preprocessing
step implemented in preprocess.py.

