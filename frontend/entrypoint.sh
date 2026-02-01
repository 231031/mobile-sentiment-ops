#!/bin/sh
set -e

envsubst < config.template.js > dist/config.js

exec "$@"
