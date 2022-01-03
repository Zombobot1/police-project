FROM ubuntu:latest
RUN apt-get update -y 
RUN apt-get install  -y python3.9
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev
RUN apt-get install -y  tesseract-ocr-deu
RUN apt-get install -y  python3-pip
RUN apt-get install -y curl
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH="${PATH}:/root/.poetry/bin"
RUN  pip install uvicorn[standard]
RUN pip install fastapi
RUN mkdir /app
WORKDIR /app
COPY pyproject.toml .
RUN poetry install
COPY . .



CMD ["poetry", "run", "start"]



