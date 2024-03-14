FROM acuvity/mldev:latest

COPY code code
COPY stage* ./
COPY Makefile Makefile
COPY pyproject.toml pyproject.toml

CMD [ "sleep", "infinity" ]
