sudo rm -rf /home/tink/airfight/RTMData
sudo rm -rf logs
# python run.py
docker stop xsim_2022
docker rm xsim_2022
python run_rl_train.py