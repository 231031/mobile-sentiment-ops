#!/bin/bash
set -e

# Create Label Studio Database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE lsdb;
    GRANT ALL PRIVILEGES ON DATABASE lsdb TO "$POSTGRES_USER";
EOSQL