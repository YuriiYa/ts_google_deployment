FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install scikit-learn
RUN pip3 install flask
RUN pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install transformers

EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]


