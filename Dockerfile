FROM continuumio/miniconda3

WORKDIR /Fusemachines-AI-Training

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "Fusemachines-AI-Training", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
COPY . .

# Give permission to run script and download dataset
RUN chmod +x make_dataset.sh
RUN ./make_dataset.sh

ENTRYPOINT ["conda", "run", "-n", "Fusemachines-AI-Training", "python", "app.py"]