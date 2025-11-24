#!/bin/sh

# Substitute environment variables in config.template.js and save to dist/config.js
envsubst < config.template.js > dist/config.js

# Execute the CMD
exec "$@"
