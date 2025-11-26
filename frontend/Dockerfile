FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN chmod +x node_modules/typescript/bin/tsc

RUN npm run build

# ───────────── Production stage ──────────────
FROM node:18-alpine AS runner

WORKDIR /app

RUN apk add --no-cache gettext

# Install serve and tools
RUN npm install -g serve && apk add --no-cache dos2unix

# Copy built assets and scripts
COPY --from=builder /app/dist ./dist
COPY config.template.js ./config.template.js
COPY entrypoint.sh ./entrypoint.sh

# Fix line endings and permissions
RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

# Runtime ENV config via entrypoint
ENTRYPOINT ["./entrypoint.sh"]
CMD ["serve", "-s", "dist", "-l", "5173"]

EXPOSE 5173
