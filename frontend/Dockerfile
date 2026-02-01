FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN chmod +x node_modules/typescript/bin/tsc
RUN npm run build

FROM node:18-alpine AS runner

WORKDIR /app
RUN apk add --no-cache gettext && npm install -g serve

COPY --from=builder /app/dist ./dist
COPY entrypoint.sh ./
COPY config.template.js ./

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["serve", "-s", "dist", "-l", "5173"]

EXPOSE 5173