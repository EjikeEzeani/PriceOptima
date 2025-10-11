/** @type {import('next').NextConfig} */
const nextConfig = {
  // Force static export for Vercel
  output: 'export',
  trailingSlash: true,
  distDir: 'out',
  
  // Disable server-side features for static export
  images: {
    unoptimized: true
  },
  
  // Disable problematic features
  eslint: {
    ignoreDuringBuilds: true
  },
  typescript: {
    ignoreBuildErrors: true
  },
  
  // Minimal experimental features
  experimental: {
    optimizeCss: false
  },
  
  // Webpack configuration for static build
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        crypto: false,
        stream: false,
        util: false,
        buffer: false,
        process: false
      }
    }
    return config
  }
}

export default nextConfig