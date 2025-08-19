
helm upgrade --install experts ./experts-chart
uvicorn app:app --reload --port 8000
