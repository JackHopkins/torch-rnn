cd data 
SECONDS=0
echo "Beginning reversal of file $1"
tail -r $1 | rev | tr '[:lower:]' '[:upper:]' > reversed-$1
#tr '[:lower:]' '[:upper:]' < reversed-$1 | cat reversed-$1
duration=$SECONDS
echo "File reversed in $(($duration / 60)) minutes and $(($duration % 60)) seconds"
echo "Beginning processing of $1"
SECONDS=0
python ../scripts/preprocess.py --input_txt reversed-$1 --output_h5 $2.h5 --output_json $2.json 
duration=$SECONDS
echo "File processed in $(($duration / 60)) minutes and $(($duration % 60)) seconds"
cd .. 
echo "Beginning to train LSTM"
echo "Epochs: $4"
echo "RNN Size: $3"
echo "Learning rate: $5"
th train.lua -input_h5 ./data/$2.h5 -input_json ./data/$2.json -rnn_size $3 -max_epochs $4 -learning_rate $5 -gpu_backend opencl