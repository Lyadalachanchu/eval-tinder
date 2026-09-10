FROM node:22-alpine
RUN corepack enable && corepack prepare pnpm@10 --activate
WORKDIR /app
COPY package.json pnpm-lock.yaml .npmrc ./
RUN pnpm install --frozen-lockfile
COPY . .
# Vite bakes VITE_API_BASE into the bundle at build time. Pass it as a build
# arg (docker build --build-arg VITE_API_BASE=http://api:8000); it can also be
# overridden at runtime from the in-app settings panel.
ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=$VITE_API_BASE
RUN pnpm build
EXPOSE 5173
CMD ["pnpm", "exec", "vite", "preview", "--host", "0.0.0.0", "--port", "5173"]
