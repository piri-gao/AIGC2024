start_year=2022
increment=1
max_year=2030

sudo rm -rf /home/tink/airfight/RTMData
sudo rm -rf logs/*

for ((year=start_year; year<=max_year; year+=increment))
do
    container_name="xsim_$year"
    docker stop $container_name
    docker rm $container_name
done

# python run.py
python run_rl_train.py