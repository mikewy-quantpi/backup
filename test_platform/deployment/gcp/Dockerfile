FROM python:3.11-slim

WORKDIR /tmp/test-framework

ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /tmp/test-framework 
ENV PICRYSTAL_ROOT_PATH /tmp/test-framework

RUN apt-get update && apt-get install -y build-essential

RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

COPY picrystal_test /tmp/test-framework/picrystal_test
COPY setup.py /tmp/test-framework/setup.py
RUN pip install -e /tmp/test-framework 

# COPY picrystal /tmp/test-framework/picrystal

WORKDIR /tmp/test-framework/picrystal_test/test_platform
RUN pip install -r requirements.txt

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "80"]
