FROM huaweinoah/smarts:v0.4.15

WORKDIR /marl

ADD . .

RUN pip install -e .

RUN scl scenario build scenarios/loop
