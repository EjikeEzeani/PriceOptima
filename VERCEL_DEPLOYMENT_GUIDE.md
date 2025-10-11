# 🚀 Complete Vercel Deployment Guide for PriceOptima Dashboard

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [ ] GitHub repository with your code
- [ ] Vercel account (free tier works)
- [ ] Backend deployed (Render/Railway) - **REQUIRED**
- [ ] Node.js installed locally (for testing)

## 🎯 Step-by-Step Deployment Process

### **Phase 1: Backend Deployment (Required First)**

#### **Option A: Deploy to Render (Recommended)**
1. **Go to [Render.com](https://render.com)**
2. **Connect your GitHub repository**
3. **Create a new Web Service**
4. **Configure settings:**
   - **Name**: `priceoptima-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r backend_requirements_super_minimal.txt`
   - **Start Command**: `python render_super_minimal.py`
   - **Root Directory**: Leave empty (uses root)
5. **Deploy and note the URL** (e.g., `https://priceoptima-backend.onrender.com`)

#### **Option B: Deploy to Railway**
1. **Go to [Railway.app](https://railway.app)**
2. **Connect GitHub repository**
3. **Create new project from GitHub**
4. **Configure for Python**
5. **Deploy and note the URL**

### **Phase 2: Vercel Frontend Deployment**

#### **Step 1: Prepare Your Repository**
```bash
# Ensure your repository is clean and committed
git add .
git commit -m "Prepare for Vercel deployment"
git push origin main
```

#### **Step 2: Create Vercel Project**
1. **Go to [Vercel.com](https://vercel.com)**
2. **Click "New Project"**
3. **Import your GitHub repository**
4. **Configure project settings:**

#### **Step 3: Configure Vercel Settings**
```
Framework Preset: Next.js
Root Directory: dynamic-pricing-dashboard
Build Command: npm run build
Output Directory: out
Install Command: npm install
```

#### **Step 4: Set Environment Variables**
1. **Go to Project Settings → Environment Variables**
2. **Add the following variables:**

```bash
# Required - Replace with your actual backend URL
NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com

# Optional - For debugging
PRICEOPTIMA_DEBUG=false
NODE_ENV=production
```

#### **Step 5: Deploy**
1. **Click "Deploy"**
2. **Wait for build to complete**
3. **Test your deployment**

### **Phase 3: Post-Deployment Configuration**

#### **Step 1: Verify Backend Connection**
1. **Test backend health endpoint:**
   ```
   https://your-backend-url.onrender.com/health
   ```
2. **Should return:** `{"status": "healthy"}`

#### **Step 2: Test Frontend Integration**
1. **Open your Vercel URL**
2. **Upload a test CSV file**
3. **Verify data processing works**
4. **Check all features function correctly**

## 🔧 Advanced Configuration

### **Vercel Configuration (Already Optimized)**
Your `dynamic-pricing-dashboard/vercel.json` is already configured for optimal deployment:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "out",
  "framework": "nextjs",
  "functions": {
    "app/api/**/*.ts": {
      "runtime": "nodejs18.x"
    }
  },
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        }
      ]
    }
  ],
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "https://your-backend-url.onrender.com/api/$1"
    }
  ]
}
```

### **Next.js Configuration (Already Optimized)**
Your `next.config.mjs` is already configured for static export with:
- Static export enabled
- Optimized for Vercel
- Proper webpack configuration
- Image optimization disabled for static export

## 🚨 Common Issues and Solutions

### **Issue 1: Build Fails - "No Next.js version detected"**
**Solution**: Ensure Root Directory is set to `dynamic-pricing-dashboard`

### **Issue 2: 404 Errors on Routes**
**Solution**: Verify `output: 'export'` in next.config.mjs and `trailingSlash: true`

### **Issue 3: API Calls Fail**
**Solution**: Check environment variable `NEXT_PUBLIC_API_URL` is set correctly

### **Issue 4: Images Not Loading**
**Solution**: Images are configured as `unoptimized: true` for static export

### **Issue 5: CORS Errors**
**Solution**: Backend must have proper CORS headers configured

## 📊 Testing Your Deployment

### **Local Testing Before Deploy**
```bash
# Test build locally
cd dynamic-pricing-dashboard
npm run build
npm run start

# Test with backend
# Set NEXT_PUBLIC_API_URL=http://localhost:8000
```

### **Post-Deployment Testing**
1. **Health Check**: Visit your Vercel URL
2. **File Upload**: Test CSV upload functionality
3. **EDA Analysis**: Run analysis and verify results
4. **Export Features**: Test report generation and download

## 🔄 Automated Deployment

### **Using Vercel CLI (Optional)**
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
cd dynamic-pricing-dashboard
vercel --prod
```

### **Using GitHub Actions (Advanced)**
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Vercel
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      - run: cd dynamic-pricing-dashboard && npm install && npm run build
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          working-directory: dynamic-pricing-dashboard
```

## 📈 Performance Optimization

### **Vercel Optimizations**
- ✅ Static export for fast loading
- ✅ CDN distribution globally
- ✅ Automatic HTTPS
- ✅ Image optimization (when not using static export)

### **Backend Optimizations**
- ✅ Use Render's free tier with proper scaling
- ✅ Implement caching for API responses
- ✅ Optimize database queries

## 🎉 Success Checklist

After deployment, verify:
- [ ] Frontend loads at Vercel URL
- [ ] Backend responds at Render/Railway URL
- [ ] File upload works
- [ ] EDA analysis completes
- [ ] Reports generate and download
- [ ] All charts display correctly
- [ ] No console errors in browser

## 🆘 Troubleshooting

### **If Deployment Fails**
1. Check Vercel build logs
2. Verify Root Directory setting
3. Ensure all dependencies are in package.json
4. Test build locally first

### **If Frontend Doesn't Connect to Backend**
1. Verify `NEXT_PUBLIC_API_URL` environment variable
2. Check backend is running and accessible
3. Test backend health endpoint directly
4. Check CORS configuration

### **If Features Don't Work**
1. Check browser console for errors
2. Verify API endpoints are correct
3. Test with sample data
4. Check network requests in browser dev tools

## 📞 Support

If you encounter issues:
1. Check Vercel deployment logs
2. Verify backend deployment status
3. Test locally first
4. Check environment variables
5. Review this guide step by step

---

**🎯 Key Points to Remember:**
1. **Backend must be deployed first** (Render/Railway)
2. **Root Directory must be `dynamic-pricing-dashboard`**
3. **Environment variable `NEXT_PUBLIC_API_URL` is critical**
4. **Test locally before deploying**
5. **Check build logs if deployment fails**

**Your application is ready for seamless Vercel deployment! 🚀**
