"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, FileText, CheckCircle, AlertCircle, X, Info } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"

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

  // Parse CSV file content
  const parseCSV = (content: string) => {
    const lines = content.split("\n").filter((line) => line.trim())
    if (lines.length < 2) throw new Error("File must contain at least a header row and one data row")

    const headers = lines[0].split(",").map((h) => h.trim().replace(/"/g, ""))
    const rows = lines.slice(1).map((line) => {
      const values = line.split(",").map((v) => v.trim().replace(/"/g, ""))
      const row: any = {}
      headers.forEach((header, index) => {
        row[header] = values[index] || ""
      })
      return row
    })

    return { headers, rows }
  }

  // Validate required columns
  const validateData = (headers: string[]) => {
    const requiredColumns = ["date", "product", "category", "price", "quantity", "revenue"]
    const normalizedHeaders = headers.map((h) => h.toLowerCase().replace(/[^a-z]/g, ""))

    const missingColumns = requiredColumns.filter((col) => !normalizedHeaders.some((header) => header.includes(col)))

    if (missingColumns.length > 0) {
      throw new Error(
        `Missing required columns: ${missingColumns.join(", ")}. Please ensure your data includes: Date, Product Name, Category, Price, Quantity Sold, Revenue`,
      )
    }
  }

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
      // Process first file (for demo, we'll focus on CSV)
      const file = uploadedFiles[0]

      if (!file.name.endsWith(".csv")) {
        throw new Error("For this demo, please upload a CSV file. Excel support coming soon!")
      }

      // Read file content
      const content = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = (e) => resolve(e.target?.result as string)
        reader.onerror = () => reject(new Error("Failed to read file"))
        reader.readAsText(file)
      })

      setUploadProgress(30)

      // Parse CSV
      const { headers, rows } = parseCSV(content)
      setUploadProgress(60)

      // Validate data structure
      validateData(headers)
      setUploadProgress(80)

      // Generate data summary and insights
      const processedData = {
        files: uploadedFiles.map((file) => ({
          name: file.name,
          size: file.size,
          type: file.type,
        })),
        headers,
        rows: rows.slice(0, 1000), // Limit to first 1000 rows for performance
        summary: {
          totalRecords: rows.length,
          dateRange: `${rows[0]?.date || "N/A"} to ${rows[rows.length - 1]?.date || "N/A"}`,
          products: new Set(rows.map((r) => r.product || r.Product || r["Product Name"])).size,
          categories: new Set(rows.map((r) => r.category || r.Category)).size,
          totalRevenue: rows.reduce((sum, row) => {
            const revenue = Number.parseFloat(row.revenue || row.Revenue || "0")
            return sum + (isNaN(revenue) ? 0 : revenue)
          }, 0),
          avgPrice:
            rows.reduce((sum, row) => {
              const price = Number.parseFloat(row.price || row.Price || "0")
              return sum + (isNaN(price) ? 0 : price)
            }, 0) / rows.length,
        },
        preview: rows.slice(0, 5), // First 5 rows for preview
      }

      setUploadProgress(100)
      setDataPreview(processedData)
      onDataUploaded(processedData)
      setSuccess(true)
    } catch (err: any) {
      setError(err.message || "Failed to process files. Please check your data format and try again.")
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-2">Upload Sales Data</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Upload your supermarket sales data in CSV format. The system will analyze your data to generate optimal
          pricing strategies that reduce waste and increase profitability.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Area */}
        <Card className="gradient-bg">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Upload className="h-5 w-5 text-primary" />
              <span>Data Upload</span>
            </CardTitle>
            <CardDescription>Drag and drop your sales data files or click to browse</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive ? "border-primary bg-primary/10" : "border-border hover:border-primary/50"
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              {isDragActive ? (
                <p className="text-primary font-medium">Drop the files here...</p>
              ) : (
                <div>
                  <p className="text-foreground font-medium mb-2">Drop sales data files here, or click to select</p>
                  <p className="text-sm text-muted-foreground">Supports CSV files up to 50MB each</p>
                </div>
              )}
            </div>

            {uploadedFiles.length > 0 && (
              <div className="mt-4 space-y-2">
                <h4 className="font-medium text-foreground">Uploaded Files:</h4>
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="flex items-center justify-between bg-accent/50 rounded-lg p-3">
                    <div className="flex items-center space-x-2">
                      <FileText className="h-4 w-4 text-primary" />
                      <span className="text-sm font-medium">{file.name}</span>
                      <span className="text-xs text-muted-foreground">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => removeFile(index)} className="h-8 w-8 p-0">
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            {uploadedFiles.length > 0 && (
              <Button onClick={processFiles} disabled={isProcessing} className="w-full mt-4">
                {isProcessing ? "Processing..." : "Process Files"}
              </Button>
            )}

            {isProcessing && (
              <div className="mt-4">
                <Progress value={uploadProgress} className="w-full" />
                <p className="text-sm text-muted-foreground mt-2 text-center">Processing files... {uploadProgress}%</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Requirements & Guidelines */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Info className="h-5 w-5 text-primary" />
              <span>Data Requirements</span>
            </CardTitle>
            <CardDescription>Ensure your data meets these requirements for optimal analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">Required Columns</p>
                  <p className="text-sm text-muted-foreground">
                    Date, Product Name, Category, Price, Quantity Sold, Revenue
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">Optional: Waste Amount, Cost, Supplier</p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">Date Format</p>
                  <p className="text-sm text-muted-foreground">YYYY-MM-DD, DD/MM/YYYY, or MM/DD/YYYY</p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">Data Quality</p>
                  <p className="text-sm text-muted-foreground">
                    Clean data with minimal missing values for best results
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">File Format</p>
                  <p className="text-sm text-muted-foreground">CSV format preferred, maximum 50MB per file</p>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-border">
              <h4 className="font-medium text-foreground mb-2">Sample Data Structure</h4>
              <div className="bg-muted/50 rounded-lg p-3 text-xs font-mono overflow-x-auto">
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
        <Card className="gradient-bg">
          <CardHeader>
            <CardTitle>Data Preview & Summary</CardTitle>
            <CardDescription>Overview of your uploaded data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-3 bg-primary/10 rounded-lg">
                <p className="text-2xl font-bold text-primary">{dataPreview.summary.totalRecords.toLocaleString()}</p>
                <p className="text-sm text-muted-foreground">Total Records</p>
              </div>
              <div className="text-center p-3 bg-chart-2/10 rounded-lg">
                <p className="text-2xl font-bold text-chart-2">{dataPreview.summary.products}</p>
                <p className="text-sm text-muted-foreground">Unique Products</p>
              </div>
              <div className="text-center p-3 bg-chart-3/10 rounded-lg">
                <p className="text-2xl font-bold text-chart-3">{dataPreview.summary.categories}</p>
                <p className="text-sm text-muted-foreground">Categories</p>
              </div>
              <div className="text-center p-3 bg-chart-4/10 rounded-lg">
                <p className="text-2xl font-bold text-chart-4">₦{dataPreview.summary.totalRevenue.toLocaleString()}</p>
                <p className="text-sm text-muted-foreground">Total Revenue</p>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="font-medium text-foreground">Sample Data (First 5 Rows):</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border">
                      {dataPreview.headers.slice(0, 6).map((header: string, index: number) => (
                        <th key={index} className="text-left p-2 font-medium text-muted-foreground">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dataPreview.preview.map((row: any, index: number) => (
                      <tr key={index} className="border-b border-border/50">
                        {dataPreview.headers.slice(0, 6).map((header: string, cellIndex: number) => (
                          <td key={cellIndex} className="p-2 text-foreground">
                            {row[header] || "N/A"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
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
