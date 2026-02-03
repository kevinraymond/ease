FROM node:18-alpine AS build

WORKDIR /app

# Copy dependency files first for layer caching
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code and build
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
