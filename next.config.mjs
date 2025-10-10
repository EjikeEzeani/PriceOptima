/** @type {import('next').NextConfig} */
const nextConfig = {
  // Force static export
  output: 'export',
  trailingSlash: true,
  distDir: 'out',
  // Disable server-side features
  images: {
    unoptimized: true,
    loader: 'custom',
    loaderFile: './imageLoader.js'
  },
  // Disable server-side rendering
  eslint: {
    ignoreDuringBuilds: true
  },
  typescript: {
    ignoreBuildErrors: true
  },
  // Disable API routes
  experimental: {
    optimizeCss: true,
    optimizePackageImports: ['react', 'react-dom']
  },
  // Webpack configuration for static build
  webpack: (config, { isServer }) => {
    // Disable server-side webpack
    if (isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false
      }
    }
    return config
  }
}

export default nextConfig