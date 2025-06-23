# Use Node.js 20
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application
COPY . .

# Build Next.js app
RUN npm run build

# Expose port (default for Next.js)
EXPOSE 3000

# Start Next.js app in production mode
CMD ["npm", "run", "start"]
