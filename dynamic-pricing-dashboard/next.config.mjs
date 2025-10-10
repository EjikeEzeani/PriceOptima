/** @type {import('next').NextConfig} */
const nextConfig = {
  // Disable strict mode for faster builds
  reactStrictMode: false,
  
  // Disable ESLint during builds
  eslint: {
    ignoreDuringBuilds: true,
  },
  
  // Disable TypeScript errors during builds
  typescript: {
    ignoreBuildErrors: true,
  },
  
  // Optimize images
  images: {
    unoptimized: true,
    domains: [],
  },
  
  // Disable source maps in production
  productionBrowserSourceMaps: false,
  
  // Optimize build
  experimental: {
    optimizeCss: true,
    optimizePackageImports: ['lucide-react', '@radix-ui/react-icons'],
  },
  
  // Webpack optimizations
  webpack: (config, { isServer, dev }) => {
    // Reduce bundle size
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        crypto: false,
        stream: false,
        util: false,
        buffer: false,
        process: false,
      }
    }
    
    // Optimize for production
    if (!dev) {
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
          },
        },
      }
    }
    
    return config
  },
  
  // Disable file tracing to reduce build size
  outputFileTracing: false,
  
  // API rewrites for development
  async rewrites() {
    if (process.env.NODE_ENV === 'development') {
      return [
        {
          source: '/api/:path*',
          destination: 'http://127.0.0.1:8001/:path*',
        },
      ]
    }
    return []
  },
}

export default nextConfig
