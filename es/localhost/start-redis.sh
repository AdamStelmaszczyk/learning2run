#!/usr/bin/env bash

# 1. uruchom standalone-redis.sh w wybranym przez siebie folderze (założmy, że zrobiłeś to w /home/astelma/)
# 2. aby opdalić redis'a - skopiuj start-redis.sh oraz stop-redis.sh do /home/astelma, odpal start-redis.sh będą w /home/astelma
# 3. aby wyłączyć redis'a - uruchom stop-redis.sh
nohup redis/bin/redis-server redis/redis.conf &
echo 'redis started (you can test it, just call: redis/bin/redis-cli ping)'
