#!/bin/bash
# Build optimization script for Vercel

echo "Starting optimized build process..."

# Set memory limit
export NODE_OPTIONS="--max-old-space-size=4096"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf .next
rm -rf node_modules/.cache

# Install dependencies with optimizations
echo "Installing dependencies..."
npm ci --only=production --no-audit --no-fund

# Build with optimizations
echo "Building application..."
npm run build

echo "Build completed successfully!"
