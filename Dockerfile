FROM python:3.10.12-slim
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./src /code/src

RUN python -m spacy download en_core_web_sm

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]