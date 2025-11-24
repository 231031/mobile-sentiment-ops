**Quick Start**
1. Build images and start services:
```powershell
docker-compose build
docker-compose up --force-recreate --build -d
```
3. Open UIs:
   - Airflow: http://localhost:8080
   - MLflow: http://localhost:8000

**Trigger the DAG**
- In the Airflow UI trigger the `mobile_sentiment` DAG, or run from the host:
```powershell
# find the airflow container name (example 'airflow')
docker ps
# trigger DAG by name (replace <airflow_container>)
docker exec -it <airflow_container> airflow dags trigger mobile_sentiment
```

**Notes about installation and the `lib` package**
- The DAG currently runs `pip install --no-cache-dir -e /opt/airflow/lib` before executing the pipeline. For this to work we added a minimal `setup.py` to `lib/` so it can be installed as an editable package.
- For faster, repeatable runs, install the `lib` package and any Python deps into the Airflow image instead of installing at runtime. Example Dockerfile fragment to add to your Airflow image build (recommended):
```dockerfile
# copy project code
COPY ./lib /opt/airflow/lib
COPY ./airflow/lib/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -e /opt/airflow/lib \
 && pip install --no-cache-dir -r /tmp/requirements.txt
```

**Files persisted on host**
- `./mlruns` — MLflow artifact storage
- `./mlflow.db` — MLflow sqlite DB
