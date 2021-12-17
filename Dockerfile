FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install scikit-learn
RUN pip3 install flask
RUN pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install transformers
COPY ./config.json        config.json           
COPY ./pytorch_model.bin  pytorch_model.bin  
COPY ./rng_state.pth      rng_state.pth      
COPY ./scaler.pt          scaler.pt          
COPY ./scheduler.pt       scheduler.pt       
COPY ./trainer_state.json trainer_state.json 
COPY ./training_args.bin  training_args.bin  
COPY ./ServiceForTextGeneration_docker.py  ServiceForTextGeneration_docker.py 

EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "ServiceForTextGeneration_docker.py" ]


