FROM mysql/mysql-server:latest

ENV MYSQL_ROOT_PASSWORD passw

# Каталог docker-entrypoint-initdb.d используется для хранения скриптов оболочки или SQL, которые вы хотите выполнить при первом запуске контейнера Dockerß
COPY ./init.sql /docker-entrypoint-initdb.d/