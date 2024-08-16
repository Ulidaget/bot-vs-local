FROM python:3.11
EXPOSE 8087
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT [ "streamlit", "run", "app-bot.py", "--server.port=8087", "--server.address=0.0.0.0" ]