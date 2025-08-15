FROM zys8119/node:cmd-1.0.0

RUN git clone https://github.com/zys8119/PaddleOCR.git
WORKDIR /app/PaddleOCR
RUN apt install vim -y
RUN vim app.py 
RUN apt install python3.11-venv -y
RUN python3 -m venv env
RUN env/bin/pip install -r requirements.txt
ENV PATH="env/bin:$PATH"
CMD ["python", "app.py"]
