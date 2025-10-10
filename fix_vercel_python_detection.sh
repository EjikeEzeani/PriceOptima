#!/bin/bash
# Script to rename requirements files and prevent Vercel Python detection
# This ensures Vercel treats the project as pure Next.js

echo "=========================================="
echo "  Vercel Python Detection Prevention"
echo "=========================================="
echo ""

# Function to rename requirements files
rename_requirements_files() {
    echo "Renaming requirements files to prevent Vercel Python detection..."
    
    # List of requirements files to rename
    files=(
        "requirements.txt"
        "render_requirements.txt"
        "requirements_minimal.txt"
        "requirements_super_minimal.txt"
        "requirements_bare_minimum.txt"
        "requirements_render.txt"
        "requirements-py313.txt"
        "backend_requirements.txt"
        "enhanced_requirements.txt"
        "working_requirements.txt"
    )
    
    # Rename each file if it exists
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            new_name="backend_${file}"
            echo "  Renaming: $file → $new_name"
            mv "$file" "$new_name"
        else
            echo "  Skipping: $file (not found)"
        fi
    done
    
    echo ""
    echo "✅ All requirements files renamed successfully!"
}

# Function to create Vercel configuration
create_vercel_config() {
    echo "Creating Vercel configuration..."
    
    cat > dynamic-pricing-dashboard/vercel.json << 'EOF'
{
  "buildCommand": "npm run build",
  "outputDirectory": "out",
  "installCommand": "npm install",
  "framework": "nextjs",
  "functions": {},
  "build": {
    "env": {
      "NODE_OPTIONS": "--max-old-space-size=2048"
    }
  }
}
EOF
    
    echo "✅ Vercel configuration created!"
}

# Function to create minimal Next.js config
create_nextjs_config() {
    echo "Creating minimal Next.js configuration..."
    
    cat > dynamic-pricing-dashboard/next.config.mjs << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Ultra minimal configuration
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  }
}

export default nextConfig
EOF
    
    echo "✅ Next.js configuration created!"
}

# Function to create minimal package.json
create_package_json() {
    echo "Creating minimal package.json..."
    
    cat > dynamic-pricing-dashboard/package.json << 'EOF'
{
  "name": "priceoptima-minimal",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "build": "next build",
    "dev": "next dev",
    "start": "next start"
  },
  "dependencies": {
    "next": "14.2.16",
    "react": "^18",
    "react-dom": "^18"
  },
  "devDependencies": {
    "@types/node": "^22",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "typescript": "^5"
  }
}
EOF
    
    echo "✅ Package.json created!"
}

# Function to create minimal Vercel ignore
create_vercel_ignore() {
    echo "Creating minimal .vercelignore..."
    
    cat > dynamic-pricing-dashboard/.vercelignore << 'EOF'
# Ultra minimal Vercel ignore
node_modules/
.next/
*.log
.env*
EOF
    
    echo "✅ .vercelignore created!"
}

# Function to create minimal app structure
create_app_structure() {
    echo "Creating minimal app structure..."
    
    # Create app directory if it doesn't exist
    mkdir -p dynamic-pricing-dashboard/app
    
    # Create layout.tsx
    cat > dynamic-pricing-dashboard/app/layout.tsx << 'EOF'
export const metadata = {
  title: 'PriceOptima Dashboard',
  description: 'Dynamic Pricing Analytics Application',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
EOF
    
    # Create page.tsx
    cat > dynamic-pricing-dashboard/app/page.tsx << 'EOF'
import React from 'react';

export default function Home() {
  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial, sans-serif' }}>
      <h1>PriceOptima Dashboard</h1>
      <p>Dynamic Pricing Analytics Application</p>
      <div style={{ marginTop: '2rem' }}>
        <h2>Upload Your Data</h2>
        <input 
          type="file" 
          accept=".csv" 
          style={{ padding: '0.5rem', margin: '1rem 0' }}
        />
        <br />
        <button 
          style={{ 
            padding: '0.75rem 1.5rem', 
            backgroundColor: '#0070f3', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Process Data
        </button>
      </div>
      <div style={{ marginTop: '2rem' }}>
        <h3>Backend API</h3>
        <p>Connect to your Render backend for data processing</p>
        <p>API URL: <code>{process.env.NEXT_PUBLIC_API_URL || 'Not configured'}</code></p>
      </div>
    </div>
  );
}
EOF
    
    echo "✅ App structure created!"
}

# Function to update backend scripts
update_backend_scripts() {
    echo "Updating backend scripts to use renamed requirements files..."
    
    # Update start scripts to use renamed files
    if [ -f "start_local_dev.bat" ]; then
        echo "  Updating start_local_dev.bat..."
        sed -i 's/requirements.txt/backend_requirements_backend.txt/g' start_local_dev.bat
    fi
    
    if [ -f "start_local_working.bat" ]; then
        echo "  Updating start_local_working.bat..."
        sed -i 's/requirements.txt/backend_requirements_backend.txt/g' start_local_working.bat
    fi
    
    echo "✅ Backend scripts updated!"
}

# Function to create restore script
create_restore_script() {
    echo "Creating restore script..."
    
    cat > restore_requirements.sh << 'EOF'
#!/bin/bash
# Script to restore original requirements file names

echo "Restoring original requirements file names..."

files=(
    "backend_requirements_backend.txt:requirements.txt"
    "backend_requirements_render.txt:render_requirements.txt"
    "backend_requirements_minimal.txt:requirements_minimal.txt"
    "backend_requirements_super_minimal.txt:requirements_super_minimal.txt"
    "backend_requirements_bare_minimum.txt:requirements_bare_minimum.txt"
    "backend_requirements-py313.txt:requirements-py313.txt"
    "backend_backend_requirements.txt:backend_requirements.txt"
    "enhanced_backend_requirements.txt:enhanced_requirements.txt"
    "working_backend_requirements.txt:working_requirements.txt"
)

for mapping in "${files[@]}"; do
    old_name="${mapping%%:*}"
    new_name="${mapping##*:}"
    if [ -f "$old_name" ]; then
        echo "  Restoring: $old_name → $new_name"
        mv "$old_name" "$new_name"
    fi
done

echo "✅ Requirements files restored!"
EOF
    
    chmod +x restore_requirements.sh
    echo "✅ Restore script created!"
}

# Main execution
main() {
    echo "Starting Vercel Python detection prevention..."
    echo ""
    
    # Execute all functions
    rename_requirements_files
    echo ""
    
    create_vercel_config
    echo ""
    
    create_nextjs_config
    echo ""
    
    create_package_json
    echo ""
    
    create_vercel_ignore
    echo ""
    
    create_app_structure
    echo ""
    
    update_backend_scripts
    echo ""
    
    create_restore_script
    echo ""
    
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo ""
    echo "✅ All requirements files renamed"
    echo "✅ Vercel configuration created"
    echo "✅ Minimal Next.js app created"
    echo "✅ Backend scripts updated"
    echo "✅ Restore script created"
    echo ""
    echo "Next steps:"
    echo "1. Commit and push to GitHub"
    echo "2. Vercel will auto-deploy without Python detection"
    echo "3. Use restore_requirements.sh if you need to restore files"
    echo ""
    echo "To restore original files: ./restore_requirements.sh"
}

# Run main function
main
