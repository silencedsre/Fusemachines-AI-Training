FROM python:3.8

EXPOSE 5000

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

RUN chmod +x make_dataset.sh
RUN ./make_dataset.sh

CMD ["python","app.py"]




#FROM continuumio/miniconda3
#
#WORKDIR /Fusemachines-AI-Training
#
## Create the environment:
#COPY environment.yml .
#RUN conda env create -f environment.yml
#
## Make RUN commands use the new environment:
#SHELL ["conda", "run", "-n", "Fusemachines-AI-Training", "/bin/bash", "-c"]
#
## Make sure the environment is activated:
#RUN echo "Make sure flask is installed:"
#RUN python -c "import flask"
#
##RUN apt-get install -y gnupg
##RUN apt-get install -y wget
##RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | apt-key add -
##RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" |  tee /etc/apt/sources.list.d/mongodb-org-4.2.list
##RUN apt-get update
##RUN apt-get install -y mongodb-org
##CMD systemctl start mongod
#
## The code to run when container is started:
#COPY . .
#
## Give permission to run script and download dataset
#RUN chmod +x make_dataset.sh
#RUN ./make_dataset.sh
#
#EXPOSE 5000
#ENTRYPOINT ["conda", "run", "-n", "Fusemachines-AI-Training", "python", "app.py"]