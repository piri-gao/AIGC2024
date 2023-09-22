# AIGC2022
#### 胜率
https://alidocs.dingtalk.com/i/nodes/yZvMRzlLwOAWryRErn2A8njY02pBqGox?iframeQuery=sheet_range%3Dkgqie6hm_16_3_1_1
#### 安装教程
```python
pip install -r requirements.txt
```
#### 使用说明

对于不同对手的对打配置在config.py内

```python
# 运行普通版本时，将clean_run.sh内的# python run.py注释打开，同时注释掉python run_rl_train.py

bash clean_run.sh
```
```python
# 运行RL学习型AI版本时，将clean_run.sh内的python run.py注释掉，同时反注释# python run_rl_train.py

bash clean_run.sh
```
