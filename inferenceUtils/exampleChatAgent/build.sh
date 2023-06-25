aws ecr get-login-password --region us-east-1 |sudo docker login --username AWS --password-stdin 134622832812.dkr.ecr.us-east-1.amazonaws.com
sudo docker build --no-cache --tag 134622832812.dkr.ecr.us-east-1.amazonaws.com/inference_model:latest .
sudo docker push 134622832812.dkr.ecr.us-east-1.amazonaws.com/inference_model:latest