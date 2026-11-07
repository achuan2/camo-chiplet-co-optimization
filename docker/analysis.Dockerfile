FROM moham:local

WORKDIR /workspace

COPY docker/analysis-requirements.txt /tmp/analysis-requirements.txt
RUN pip install --no-cache-dir -r /tmp/analysis-requirements.txt

ENTRYPOINT ["python3"]
