FROM python:3.11-slim

WORKDIR /app

RUN pip install uv==0.7.2

COPY pyproject.toml .
RUN uv sync --no-dev

COPY src/ src/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/erag/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
