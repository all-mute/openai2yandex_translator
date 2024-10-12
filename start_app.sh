# start_app.sh
nohup uvicorn app.main:app --host 0.0.0.0 --port 9041 &
