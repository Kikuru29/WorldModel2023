
while true
do

# 年月日時分秒を取得
DATE=`date '+%Y%m%d-%H%M'`
echo $DATE



cd ./simple_adversary_v3/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_adversary_v3.log
cd ../

cd ./simple_crypto_v3/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_crypto_v3.log
cd ../

cd ./simple_push_v3/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_push_v3.log
cd ../

cd ./simple_reference_v3/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_reference_v3.log
cd ../

cd ./simple_speaker_listener_v4/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_speaker_listener_v4.log
cd ../

cd ./simple_spread_v3/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_spread_v3.log
cd ../

cd ./simple_tag_v3/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_tag_v3.log
cd ../

cd ./simple_v3/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_v3.log
cd ../

cd ./simple_world_comm_v3/
python3 ./script-1-train.py -dt $DATE -p "./parameters.json" | tee -a simple_world_comm_v3.log
cd ../


done
