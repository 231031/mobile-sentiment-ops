FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# -----------------------------------------------
# 1. Accept the argument from Docker Compose
ARG VITE_API_URL
# 2. Set it as an ENV var so Vite can see it during build
ENV VITE_API_URL=$VITE_API_URL
# -----------------------------------------------

RUN chmod +x node_modules/typescript/bin/tsc
# When this runs, Vite reads the ENV above and hardcodes it into the JS
RUN npm run build

# ───────────── Production stage ──────────────
FROM node:18-alpine AS runner

WORKDIR /app

# We don't need gettext/dos2unix anymore because the URL is already inside the files
RUN npm install -g serve

# Copy built assets
COPY --from=builder /app/dist ./dist

# Simple command
CMD ["serve", "-s", "dist", "-l", "5173"]

EXPOSE 5173