#!/bin/bash

set -e

wget --quiet http://download.redis.io/releases/redis-3.2.7.tar.gz -O redis-3.2.7.tar.gz
tar -xvzf redis-3.2.7.tar.gz
mkdir redis
cd redis-3.2.7
make
make install PREFIX="../../redis"
cp redis.conf ../redis/
cd ..
mkdir redis/wd/
sed -ie 's/dir \.\//dir redis\/wd/' redis/redis.conf
echo 'save ""' | tee -a redis/redis.conf

echo "unixsocket $(pwd)/redis/redis.sock" | tee -a redis/redis.conf
echo "unixsocketperm 777" | tee -a redis/redis.conf

rm -rf redis-3.2.7 redis-3.2.7.tar.gz
