/**
 * API Configuration and Client
 * Centralized API endpoints and data fetching utilities
 */

// Prefer loopback IP to avoid IPv6/host resolution edge cases on Windows
// Use Next.js proxy (`/api`) by default in the browser to avoid CORS during local dev
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== 'undefined' ? '/api' : 'http://127.0.0.1:8000');

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

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
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

    try {
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
      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Upload data (with fallbacks for alternate backends)
  async uploadData(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    // Primary endpoint used by the working backends
    try {
      return await this.request<UploadResponse>('/upload', {
        method: 'POST',
        headers: {}, // Let browser set Content-Type for FormData
        body: formData,
      });
    } catch (primaryError) {
      // Fallback 1: older alias supported by some backends
      try {
        const resp = await this.request<any>('/upload-dataset', {
          method: 'POST',
          headers: {},
          body: formData,
        });

        // Normalize older responses (e.g., from app.py) to UploadResponse
        const headers: string[] = Array.isArray(resp?.headers)
          ? resp.headers
          : Array.isArray(resp?.summary?.columnNames)
            ? resp.summary.columnNames
            : Array.isArray(resp?.preview) && resp.preview.length > 0
              ? Object.keys(resp.preview[0])
              : [];

        const rows: any[] = Array.isArray(resp?.rows)
          ? resp.rows
          : Array.isArray(resp?.preview)
            ? resp.preview
            : [];

        const uploadResponse: UploadResponse = {
          files: resp?.files ?? [],
          headers,
          rows: rows.slice(0, 1000),
          summary: {
            totalRecords: Number(resp?.summary?.totalRecords ?? rows.length ?? 0),
            dateRange: 'N/A',
            products: 0,
            categories: 0,
            totalRevenue: 0,
            avgPrice: 0,
          },
          preview: Array.isArray(resp?.preview) ? resp.preview : rows.slice(0, 5),
          totalRows: Number(resp?.totalRows ?? rows.length ?? 0),
        };

        return uploadResponse;
      } catch (fallbackError) {
        // Re-throw original error with context
        throw primaryError;
      }
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

  // Health Check
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return response.ok;
    } catch {
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



