/** @type {import('next').NextConfig} */
const nextConfig = {
  // Minimal configuration for Vercel
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // Disable all optimizations that might cause issues
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  // No webpack customizations
  // No experimental features
  // No rewrites
}

export default nextConfig
