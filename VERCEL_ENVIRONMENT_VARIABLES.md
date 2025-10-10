# Vercel Environment Variables for Render Backend

## 🎯 Required Environment Variables for Vercel

### **Primary Environment Variable:**
```bash
NEXT_PUBLIC_API_URL=https://your-render-app-name.onrender.com
```

### **Replace `your-render-app-name` with your actual Render service name**

## 📋 Step-by-Step Setup:

### **1. Get Your Render Backend URL:**
- Go to your Render dashboard
- Find your backend service
- Copy the URL (it looks like: `https://priceoptima-1.onrender.com`)

### **2. Set Vercel Environment Variable:**

#### **Option A: Via Vercel Dashboard:**
1. Go to your Vercel project dashboard
2. Click on **Settings** → **Environment Variables**
3. Add new variable:
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: `https://your-render-app-name.onrender.com`
   - **Environment**: Production, Preview, Development (select all)

#### **Option B: Via Vercel CLI:**
```bash
vercel env add NEXT_PUBLIC_API_URL
# Enter: https://your-render-app-name.onrender.com
```

### **3. Example Configuration:**

#### **If your Render service is named "priceoptima-1":**
```bash
NEXT_PUBLIC_API_URL=https://priceoptima-1.onrender.com
```

#### **If your Render service is named "priceoptima-backend":**
```bash
NEXT_PUBLIC_API_URL=https://priceoptima-backend.onrender.com
```

## 🔧 Additional Optional Environment Variables:

### **For Debugging (Optional):**
```bash
PRICEOPTIMA_DEBUG=true
```

### **For Development (Optional):**
```bash
NODE_ENV=production
```

## ✅ Verification Steps:

### **1. Check Your Render Service:**
- Ensure your Render backend is deployed and running
- Test the health endpoint: `https://your-render-app-name.onrender.com/health`

### **2. Test Vercel Environment:**
- After setting the environment variable, redeploy Vercel
- Check the frontend logs to ensure it's connecting to the correct backend

### **3. Test Full Integration:**
- Upload a file through the Vercel frontend
- Verify it reaches the Render backend
- Check that processing completes successfully

## 🚨 Common Issues:

### **Issue 1: Wrong URL Format**
- ❌ Wrong: `http://your-render-app-name.onrender.com`
- ✅ Correct: `https://your-render-app-name.onrender.com`

### **Issue 2: Missing Protocol**
- ❌ Wrong: `your-render-app-name.onrender.com`
- ✅ Correct: `https://your-render-app-name.onrender.com`

### **Issue 3: Wrong Variable Name**
- ❌ Wrong: `API_URL` or `BACKEND_URL`
- ✅ Correct: `NEXT_PUBLIC_API_URL`

## 📊 Current Frontend Configuration:

The frontend is already configured to use this environment variable:

```typescript
// In dynamic-pricing-dashboard/lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';
```

## 🎉 Expected Result:

Once configured correctly:
- ✅ Vercel frontend will connect to Render backend
- ✅ File uploads will be processed by Render
- ✅ Results will be displayed in Vercel frontend
- ✅ Full end-to-end functionality working

## 🔄 After Setting Environment Variables:

1. **Redeploy Vercel**: Trigger a new deployment
2. **Test Connection**: Verify frontend connects to backend
3. **Monitor Logs**: Check both Vercel and Render logs
4. **Verify Functionality**: Test file upload and processing
