# Use an official Node runtime as a parent image
FROM node:14

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages
RUN npm install --legacy-peer-deps --silent

# Build  the react app
RUN npm run build

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Run node server.js when the container launches
CMD ["node", "server.js"]
