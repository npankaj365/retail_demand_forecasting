# retail_demand_forecasting
Retail Demand Forecasting

## To Run in Host System

pip install -r requirements.txt

python Model.py <class_number>


## To Run in Dockerized Container

docker build -t forecasting .

docker run -ti forecasting /bin/bash

python Model.py <class_number>
