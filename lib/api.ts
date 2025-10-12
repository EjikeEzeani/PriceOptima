/**
 * API Configuration and Client
 * Centralized API endpoints and data fetching utilities
 */

// API Base URL configuration with fallback support
const getApiBaseUrl = () => {
  // If NEXT_PUBLIC_API_BASE_URL is set, use it (for production)
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL;
  }
  
  // In browser (client-side), use direct backend URL in development, direct URL in production
  if (typeof window !== 'undefined') {
    return process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:8001' : 'https://priceoptima.onrender.com';
  }
  
  // Server-side rendering fallback
  return process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:8001' : 'https://priceoptima.onrender.com';
};

// Primary and fallback API URLs for better reliability
const PRIMARY_API_URL = getApiBaseUrl();
const FALLBACK_API_URLS = [
  'https://priceoptima.onrender.com',
  'https://priceoptima-backend.onrender.com',
  'https://priceoptima-1.onrender.com'
];

const API_BASE_URL = PRIMARY_API_URL;

export interface UploadResponse {
  files: Array<{
    name: string;
    size: number;
    type: string;
  }>;
  headers: string[];
  rows: any[];
  summary: {
    totalRecords: number;
    dateRange: string;
    products: number;
    categories: number;
    totalRevenue: number;
    avgPrice: number;
  };
  preview: any[];
  totalRows: number;
}

export interface EDAResponse {
  overview: {
    category_distribution: Record<string, number>;
    revenue_vs_waste: {
      revenue: number[];
      waste: number[];
    };
  };
  trends: {
    sales_over_time: number[];
  };
  correlations: {
    price_quantity: number;
    price_revenue: number;
  };
  insights: string[];
  recommendations: string[];
}

export interface MLResponse {
  modelId: string;
  metrics: {
    r2: number;
    rmse: number;
    mae: number;
  };
  predictions: Array<{
    actual: number;
    predicted: number;
    product: string;
  }>;
  featureImportance: Array<{
    feature: string;
    importance: number;
  }>;
}

export interface RLResponse {
  algorithm: string;
  finalReward: number;
  convergenceEpisode: number;
  policy: {
    wasteReduction: number;
    profitIncrease: number;
    customerSatisfaction: number;
  };
  trainingCurve: Array<{
    episode: number;
    reward: number;
  }>;
}

export interface ExportResponse {
  status: string;
  exported: string[];
  files: string[];
  message: string;
}

export interface BackendStatus {
  uploaded: boolean;
  processed: boolean;
  eda_complete: boolean;
  timestamp: string;
}

class APIClient {
  private baseURL: string;
  private fallbackURLs: string[];
  private isHealthy: boolean = true;
  private lastHealthCheck: number = 0;
  private healthCheckInterval: number = 30000; // 30 seconds

  constructor(baseURL: string = API_BASE_URL, fallbackURLs: string[] = FALLBACK_API_URLS) {
    this.baseURL = baseURL;
    this.fallbackURLs = fallbackURLs;
  }

  private async checkHealth(): Promise<boolean> {
    const now = Date.now();
    if (now - this.lastHealthCheck < this.healthCheckInterval) {
      return this.isHealthy;
    }

    try {
      const response = await fetch(`${this.baseURL}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      this.isHealthy = response.ok;
      this.lastHealthCheck = now;
      
      if (!this.isHealthy) {
        console.warn(`Primary API ${this.baseURL} is unhealthy`);
      }
      
      return this.isHealthy;
    } catch (error) {
      console.warn(`Health check failed for ${this.baseURL}:`, error);
      this.isHealthy = false;
      this.lastHealthCheck = now;
      return false;
    }
  }

  private async findHealthyEndpoint(): Promise<string> {
    // Check primary endpoint first
    if (await this.checkHealth()) {
      return this.baseURL;
    }

    // Try fallback endpoints
    for (const fallbackUrl of this.fallbackURLs) {
      try {
        const response = await fetch(`${fallbackUrl}/health`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          signal: AbortSignal.timeout(5000)
        });
        
        if (response.ok) {
          console.log(`Using fallback API: ${fallbackUrl}`);
          return fallbackUrl;
        }
      } catch (error) {
        console.warn(`Fallback API ${fallbackUrl} is also unhealthy:`, error);
      }
    }

    // If all endpoints fail, return primary (will show error to user)
    console.error('All API endpoints are unhealthy');
    return this.baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const healthyUrl = await this.findHealthyEndpoint();
    const url = `${healthyUrl}${endpoint}`;
    
    // Only set Content-Type for non-FormData requests
    const defaultHeaders: Record<string, string> = {};
    if (!(options.body instanceof FormData)) {
      defaultHeaders['Content-Type'] = 'application/json';
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    // Retry logic for failed requests
    const maxRetries = 3;
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`API request attempt ${attempt}/${maxRetries} to ${url}`);
        const response = await fetch(url, config);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.error || 
            errorData.detail || 
            `HTTP ${response.status}: ${response.statusText}`
          );
        }

        // EDA may return numeric arrays; always parse JSON
        const result = await response.json();
        console.log(`API request successful to ${url}`);
        return result;
      } catch (error) {
        lastError = error as Error;
        console.error(`API request attempt ${attempt}/${maxRetries} failed for ${url}:`, error);
        
        // If this is the last attempt, throw the error
        if (attempt === maxRetries) {
          break;
        }
        
        // Wait before retrying (exponential backoff)
        const delay = Math.pow(2, attempt - 1) * 1000;
        console.log(`Waiting ${delay}ms before retry...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    // Enhanced error message for better debugging
    const errorMessage = lastError?.message || 'Unknown error';
    throw new Error(`API request failed after ${maxRetries} attempts. Last error: ${errorMessage}. Please check if the backend service is running.`);
  }

  // Upload data (with enhanced error handling and timeout)
  async uploadData(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    // Find healthy endpoint for upload
    const healthyUrl = await this.findHealthyEndpoint();

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000); // 60s timeout

    try {
      console.log(`Uploading file: ${file.name} (${file.size} bytes) to ${healthyUrl}/upload`);
      
      const response = await fetch(`${healthyUrl}/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        headers: {
          // Don't set Content-Type for FormData - let browser set it with boundary
        },
      });
      
      clearTimeout(timeout);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Upload failed: ${response.status} ${response.statusText}`, errorText);
        throw new Error(`Upload failed: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      console.log('Upload successful:', result);
      return result;
    } catch (error) {
      clearTimeout(timeout);
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('Upload timeout after 60 seconds');
          throw new Error('Upload timeout - file may be too large or server is slow');
        }
        console.error('Upload error:', error.message);
        throw error;
      }
      
      console.error('Unknown upload error:', error);
      throw new Error('Network error during upload - please check your connection');
    }
  }

  // EDA Analysis
  async runEDA(): Promise<EDAResponse> {
    return this.request<EDAResponse>('/eda', {
      method: 'POST',
      body: JSON.stringify({}),
    });
  }

  // ML Training
  async trainML(model: string): Promise<MLResponse> {
    return this.request<MLResponse>('/ml', {
      method: 'POST',
      body: JSON.stringify({ model }),
    });
  }

  // RL Simulation
  async runRL(algorithm: string): Promise<RLResponse> {
    return this.request<RLResponse>('/rl', {
      method: 'POST',
      body: JSON.stringify({ algorithm }),
    });
  }

  // Export Reports
  async exportReports(items: string[]): Promise<ExportResponse> {
    return this.request<ExportResponse>('/export', {
      method: 'POST',
      body: JSON.stringify({ items }),
    });
  }

  // Download File
  async downloadFile(filename: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/download/${filename}`);
    
    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }
    
    return response.blob();
  }

  // Health Check with detailed logging
  async healthCheck(): Promise<boolean> {
    try {
      const healthyUrl = await this.findHealthyEndpoint();
      console.log(`Health check: Testing connection to ${healthyUrl}/health`);
      const response = await fetch(`${healthyUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(10000) // 10 second timeout
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Health check successful:', data);
        return true;
      } else {
        console.error(`Health check failed: HTTP ${response.status} ${response.statusText}`);
        return false;
      }
    } catch (error) {
      console.error('Health check error:', error);
      return false;
    }
  }

  // Backend Status (fallback to /results on older backends)
  async getStatus(): Promise<BackendStatus> {
    try {
      return await this.request<BackendStatus>('/status', {
        method: 'GET',
      });
    } catch (e) {
      try {
        const resp = await this.request<any>('/results', { method: 'GET' });
        const mapped: BackendStatus = {
          uploaded: Boolean(resp?.uploaded ?? resp?.data_uploaded ?? false),
          processed: Boolean(resp?.processed ?? resp?.data_uploaded ?? false),
          eda_complete: Boolean(resp?.eda_complete ?? resp?.eda_completed ?? false),
          timestamp: new Date().toISOString(),
        };
        return mapped;
      } catch {
        return {
          uploaded: false,
          processed: false,
          eda_complete: false,
          timestamp: new Date().toISOString(),
        };
      }
    }
  }
}

// Export singleton instance
export const apiClient = new APIClient();

// Export individual functions for convenience
export const uploadData = (file: File) => apiClient.uploadData(file);
export const runEDA = () => apiClient.runEDA();
export const trainML = (model: string) => apiClient.trainML(model);
export const runRL = (algorithm: string) => apiClient.runRL(algorithm);
export const exportReports = (items: string[]) => apiClient.exportReports(items);
export const downloadFile = (filename: string) => apiClient.downloadFile(filename);
export const healthCheck = () => apiClient.healthCheck();
export const getStatus = () => apiClient.getStatus();



