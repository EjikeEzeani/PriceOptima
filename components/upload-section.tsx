"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, FileText, CheckCircle, AlertCircle, X, Info } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { uploadData, type UploadResponse } from "@/lib/api"

interface UploadSectionProps {
  onDataUploaded: (data: any) => void
}

export function UploadSection({ onDataUploaded }: UploadSectionProps) {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [dataPreview, setDataPreview] = useState<any>(null)

  // Client-side validation removed - backend handles all processing for better performance

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError(null)
    setSuccess(false)
    setDataPreview(null)

    // Validate file types
    const validFiles = acceptedFiles.filter(
      (file) =>
        file.type === "text/csv" ||
        file.type === "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" ||
        file.type === "application/vnd.ms-excel" ||
        file.name.endsWith(".csv"),
    )

    if (validFiles.length !== acceptedFiles.length) {
      setError("Please upload only CSV or Excel files (.csv, .xlsx, .xls)")
      return
    }

    // Check file size (max 50MB)
    const oversizedFiles = validFiles.filter((file) => file.size > 50 * 1024 * 1024)
    if (oversizedFiles.length > 0) {
      setError("File size must be less than 50MB")
      return
    }

    setUploadedFiles((prev) => [...prev, ...validFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
    },
    multiple: true,
  })

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
    setDataPreview(null)
    setSuccess(false)
  }

  const processFiles = async () => {
    if (uploadedFiles.length === 0) return

    setIsProcessing(true)
    setUploadProgress(0)
    setError(null)

    try {
      // Process first file
      const file = uploadedFiles[0]

      if (!file.name.endsWith(".csv")) {
        throw new Error("For this demo, please upload a CSV file. Excel support coming soon!")
      }

      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 15, 85))
      }, 200)

      console.log('Starting file upload...', {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type
      })

      // Upload file directly to backend - no client-side processing
      const processedData: UploadResponse = await uploadData(file)
      
      clearInterval(progressInterval)
      setUploadProgress(100)
      
      console.log('Upload Results:', processedData)
      console.log('Preview data:', processedData.preview)
      console.log('Headers:', processedData.headers)
      console.log('Summary:', processedData.summary)

      // Validate the response
      if (!processedData || !processedData.preview || !processedData.headers) {
        throw new Error("Invalid response from server. Please try again.")
      }

      setDataPreview(processedData)
      onDataUploaded(processedData)
      setSuccess(true)
    } catch (err: any) {
      console.error('Upload error:', err)
      setError(err.message || "Failed to process files. Please check your data format and try again.")
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="space-y-4 md:space-y-6">
      <div className="text-center px-4 md:px-0">
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-2">Upload Sales Data</h2>
        <p className="text-sm md:text-base text-muted-foreground max-w-2xl mx-auto">
          Upload your supermarket sales data in CSV format. The system will analyze your data to generate optimal
          pricing strategies that reduce waste and increase profitability.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6">
        {/* Upload Area */}
        <Card className="gradient-bg mobile-card">
          <CardHeader className="p-4 md:p-6">
            <CardTitle className="flex items-center space-x-2 text-lg md:text-xl">
              <Upload className="h-4 w-4 md:h-5 md:w-5 text-primary" />
              <span>Data Upload</span>
            </CardTitle>
            <CardDescription className="text-sm md:text-base">Drag and drop your sales data files or click to browse</CardDescription>
          </CardHeader>
          <CardContent className="p-4 md:p-6">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-6 md:p-8 text-center cursor-pointer transition-colors touchable ${
                isDragActive ? "border-primary bg-primary/10" : "border-border hover:border-primary/50"
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="h-8 w-8 md:h-12 md:w-12 text-muted-foreground mx-auto mb-3 md:mb-4" />
              {isDragActive ? (
                <p className="text-primary font-medium text-sm md:text-base">Drop the files here...</p>
              ) : (
                <div>
                  <p className="text-foreground font-medium mb-2 text-sm md:text-base">Drop sales data files here, or click to select</p>
                  <p className="text-xs md:text-sm text-muted-foreground">Supports CSV files up to 50MB each</p>
                </div>
              )}
            </div>

            {uploadedFiles.length > 0 && (
              <div className="mt-4 space-y-2">
                <h4 className="font-medium text-foreground text-sm md:text-base">Uploaded Files:</h4>
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="flex items-center justify-between bg-accent/50 rounded-lg p-2 md:p-3">
                    <div className="flex items-center space-x-2 min-w-0 flex-1">
                      <FileText className="h-3 w-3 md:h-4 md:w-4 text-primary flex-shrink-0" />
                      <span className="text-xs md:text-sm font-medium truncate">{file.name}</span>
                      <span className="text-xs text-muted-foreground flex-shrink-0">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                    </div>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => removeFile(index)} 
                      className="h-6 w-6 md:h-8 md:w-8 p-0 touchable flex-shrink-0"
                    >
                      <X className="h-3 w-3 md:h-4 md:w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            {uploadedFiles.length > 0 && (
              <Button 
                onClick={processFiles} 
                disabled={isProcessing} 
                className="w-full mt-4 mobile-btn-base touchable"
              >
                {isProcessing ? "Processing..." : "Process Files"}
              </Button>
            )}

            {isProcessing && (
              <div className="mt-4">
                <Progress value={uploadProgress} className="w-full" />
                <p className="text-xs md:text-sm text-muted-foreground mt-2 text-center">Processing files... {uploadProgress}%</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Requirements & Guidelines */}
        <Card className="mobile-card">
          <CardHeader className="p-4 md:p-6">
            <CardTitle className="flex items-center space-x-2 text-lg md:text-xl">
              <Info className="h-4 w-4 md:h-5 md:w-5 text-primary" />
              <span>Data Requirements</span>
            </CardTitle>
            <CardDescription className="text-sm md:text-base">Ensure your data meets these requirements for optimal analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 md:space-y-4 p-4 md:p-6">
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <CheckCircle className="h-4 w-4 md:h-5 md:w-5 text-green-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-foreground text-sm md:text-base">Required Columns</p>
                  <p className="text-xs md:text-sm text-muted-foreground">
                    Date, Product Name, Category, Price, Quantity Sold, Revenue
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">Optional: Waste Amount, Cost, Supplier</p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <CheckCircle className="h-4 w-4 md:h-5 md:w-5 text-green-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-foreground text-sm md:text-base">Date Format</p>
                  <p className="text-xs md:text-sm text-muted-foreground">YYYY-MM-DD, DD/MM/YYYY, or MM/DD/YYYY</p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <CheckCircle className="h-4 w-4 md:h-5 md:w-5 text-green-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-foreground text-sm md:text-base">Data Quality</p>
                  <p className="text-xs md:text-sm text-muted-foreground">
                    Clean data with minimal missing values for best results
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <CheckCircle className="h-4 w-4 md:h-5 md:w-5 text-green-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-foreground text-sm md:text-base">File Format</p>
                  <p className="text-xs md:text-sm text-muted-foreground">CSV format preferred, maximum 50MB per file</p>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-border">
              <h4 className="font-medium text-foreground mb-2 text-sm md:text-base">Sample Data Structure</h4>
              <div className="bg-muted/50 rounded-lg p-2 md:p-3 text-xs font-mono overflow-x-auto mobile-table-container">
                <div className="text-muted-foreground whitespace-nowrap">
                  Date,Product,Category,Price,Quantity,Revenue
                </div>
                <div className="text-foreground mt-1 whitespace-nowrap">2024-01-01,Rice 5kg,Grains,2500,45,112500</div>
                <div className="text-foreground whitespace-nowrap">
                  2024-01-01,Tomatoes 1kg,Vegetables,800,120,96000
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Preview */}
      {dataPreview && (
        <Card className="gradient-bg mobile-card">
          <CardHeader className="p-4 md:p-6">
            <CardTitle className="text-lg md:text-xl">Data Preview & Summary</CardTitle>
            <CardDescription className="text-sm md:text-base">Overview of your uploaded data</CardDescription>
          </CardHeader>
          <CardContent className="p-4 md:p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 mb-4 md:mb-6 mobile-metrics">
              <div className="text-center p-3 md:p-4 metric-card-1 rounded-lg mobile-metric-card">
                <p className="metric-value high-contrast-text text-lg md:text-2xl">{dataPreview.summary?.totalRecords?.toLocaleString() || '0'}</p>
                <p className="metric-label high-contrast-label text-xs md:text-sm">Total Records</p>
              </div>
              <div className="text-center p-3 md:p-4 metric-card-2 rounded-lg mobile-metric-card">
                <p className="metric-value high-contrast-text text-lg md:text-2xl">{dataPreview.summary?.products || '0'}</p>
                <p className="metric-label high-contrast-label text-xs md:text-sm">Unique Products</p>
              </div>
              <div className="text-center p-3 md:p-4 metric-card-3 rounded-lg mobile-metric-card">
                <p className="metric-value high-contrast-text text-lg md:text-2xl">{dataPreview.summary?.categories || '0'}</p>
                <p className="metric-label high-contrast-label text-xs md:text-sm">Categories</p>
              </div>
              <div className="text-center p-3 md:p-4 metric-card-4 rounded-lg mobile-metric-card">
                <p className="metric-value high-contrast-text analytics-text text-lg md:text-2xl">₦{dataPreview.summary?.totalRevenue?.toLocaleString() || '0'}</p>
                <p className="metric-label high-contrast-label text-xs md:text-sm">Total Revenue</p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-foreground text-sm md:text-base">Sample Data (First 5 Rows):</h4>
                <Badge variant="secondary" className="text-xs">
                  {dataPreview.totalRows || dataPreview.preview?.length || 0} total records
                </Badge>
              </div>
              <div className="overflow-x-auto border border-border rounded-lg mobile-table-container">
                <table className="data-preview-table w-full mobile-table">
                  <thead>
                    <tr>
                      {(dataPreview.headers || []).slice(0, 6).map((header: string, index: number) => (
                        <th key={index} className="text-left p-2 md:p-3 text-xs md:text-sm">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(dataPreview.preview || []).map((row: any, index: number) => {
                      return (
                        <tr key={index}>
                          {(dataPreview.headers || []).slice(0, 6).map((header: string, cellIndex: number) => {
                            const value = row?.[header];
                            return (
                              <td key={cellIndex} className="analytics-text p-2 md:p-3 text-xs md:text-sm">
                                {(() => {
                                  if (value === null || value === undefined) return "N/A";
                                  if (typeof value === 'object') return JSON.stringify(value);
                                  return String(value);
                                })()}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {dataPreview.totalRows > 5 && (
                <p className="text-xs text-muted-foreground text-center">
                  Showing first 5 rows of {dataPreview.totalRows} total records
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Status Messages */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800 dark:text-green-200">
            <div className="space-y-2">
              <p className="font-medium">Data processed successfully!</p>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">{dataPreview.summary.totalRecords.toLocaleString()} records</Badge>
                <Badge variant="secondary">{dataPreview.summary.products} products</Badge>
                <Badge variant="secondary">{dataPreview.summary.categories} categories</Badge>
              </div>
              <p className="text-sm">You can now proceed to the EDA section to explore your data insights.</p>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
