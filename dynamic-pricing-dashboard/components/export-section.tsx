"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Download,
  FileText,
  ImageIcon,
  BarChart3,
  FileSpreadsheet,
  Presentation,
  CheckCircle,
  Package,
  AlertCircle,
} from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { exportReports, downloadFile, healthCheck as apiHealthCheck, type ExportResponse } from "@/lib/api"

interface ExportSectionProps {
  data: any
}

const exportOptions = [
  {
    id: "summary_report",
    name: "Executive Summary Report",
    description: "Comprehensive PDF report with key findings and recommendations",
    icon: FileText,
    format: "PDF",
    size: "2.3 MB",
    category: "reports",
  },
  {
    id: "technical_report",
    name: "Technical Analysis Report",
    description: "Detailed technical documentation with methodology and results",
    icon: FileText,
    format: "PDF",
    size: "8.7 MB",
    category: "reports",
  },
  {
    id: "presentation",
    name: "Presentation Slides",
    description: "PowerPoint presentation for stakeholder meetings",
    icon: Presentation,
    format: "PPTX",
    size: "12.4 MB",
    category: "presentations",
  },
  {
    id: "raw_data",
    name: "Processed Dataset",
    description: "Cleaned and processed sales data with predictions",
    icon: FileSpreadsheet,
    format: "CSV",
    size: "5.2 MB",
    category: "data",
  },
  {
    id: "ml_results",
    name: "ML Model Results",
    description: "Model predictions, metrics, and feature importance data",
    icon: BarChart3,
    format: "JSON",
    size: "1.8 MB",
    category: "data",
  },
  {
    id: "rl_policy",
    name: "RL Policy Data",
    description: "Trained reinforcement learning policy and training logs",
    icon: Package,
    format: "PKL",
    size: "3.1 MB",
    category: "data",
  },
  {
    id: "visualizations",
    name: "Charts & Visualizations",
    description: "High-resolution charts and graphs as image files",
    icon: ImageIcon,
    format: "PNG/SVG",
    size: "15.6 MB",
    category: "visualizations",
  },
]

export function ExportSection({ data }: ExportSectionProps) {
  const [selectedItems, setSelectedItems] = useState<string[]>([])
  const [isExporting, setIsExporting] = useState(false)
  const [exportProgress, setExportProgress] = useState(0)
  const [exportComplete, setExportComplete] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)
  const [exportResults, setExportResults] = useState<ExportResponse | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string>("all")

  const categories = [
    { id: "all", name: "All Items", count: exportOptions.length },
    { id: "reports", name: "Reports", count: exportOptions.filter((item) => item.category === "reports").length },
    { id: "data", name: "Data Files", count: exportOptions.filter((item) => item.category === "data").length },
    {
      id: "visualizations",
      name: "Visualizations",
      count: exportOptions.filter((item) => item.category === "visualizations").length,
    },
    {
      id: "presentations",
      name: "Presentations",
      count: exportOptions.filter((item) => item.category === "presentations").length,
    },
  ]

  const filteredOptions =
    selectedCategory === "all" ? exportOptions : exportOptions.filter((item) => item.category === selectedCategory)

  const toggleItem = (itemId: string) => {
    setSelectedItems((prev) => (prev.includes(itemId) ? prev.filter((id) => id !== itemId) : [...prev, itemId]))
  }

  const selectAll = () => {
    setSelectedItems(filteredOptions.map((item) => item.id))
  }

  const clearAll = () => {
    setSelectedItems([])
  }

  const startExport = async () => {
    if (selectedItems.length === 0) return

    setIsExporting(true)
    setExportProgress(0)
    setExportComplete(false)
    setExportError(null)
    setExportResults(null)

    try {
      console.log('Starting export with items:', selectedItems)
      
      // Test backend connection first via centralized API client
      const ok = await apiHealthCheck()
      if (!ok) throw new Error('Backend server is not responding. Please ensure the backend is running on port 8000.')

      // Simulate export progress
      for (let i = 0; i <= 100; i += 5) {
        setExportProgress(i)
        await new Promise((resolve) => setTimeout(resolve, 50))
      }

      // Call backend export API using centralized client
      const results: ExportResponse = await exportReports(selectedItems)
      console.log('Export Results:', results)
      
      // Validate results
      if (!results || !results.files || results.files.length === 0) {
        throw new Error('Export completed but no files were generated. Please try again.')
      }
      
      setExportResults(results)
      setExportComplete(true)
      console.log('Export completed successfully')
    } catch (error: any) {
      console.error('Error exporting:', error)
      setExportError(error.message || 'Export failed. Please check your connection and try again.')
      setExportComplete(false)
    } finally {
      setIsExporting(false)
    }
  }

  const getTotalSize = () => {
    return selectedItems
      .reduce((total, itemId) => {
        const item = exportOptions.find((opt) => opt.id === itemId)
        if (!item) return total
        const size = Number.parseFloat(item.size.split(" ")[0])
        return total + size
      }, 0)
      .toFixed(1)
  }

  // Note: Export section can work even without data prop since it uses backend data

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-2">Export Results</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Download comprehensive reports, data files, and visualizations from your dynamic pricing analysis for
          presentation and implementation.
        </p>
      </div>

      {/* Category Filter */}
      <Card className="gradient-bg">
        <CardHeader>
          <CardTitle>Export Categories</CardTitle>
          <CardDescription>Filter export options by category</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {categories.map((category) => (
              <Button
                key={category.id}
                variant={selectedCategory === category.id ? "default" : "outline"}
                onClick={() => setSelectedCategory(category.id)}
                className="flex items-center space-x-2"
              >
                <span>{category.name}</span>
                <Badge variant="secondary" className="ml-1">
                  {category.count}
                </Badge>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Export Options */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Available Exports</CardTitle>
                <div className="flex space-x-2">
                  <Button variant="outline" size="sm" onClick={selectAll}>
                    Select All
                  </Button>
                  <Button variant="outline" size="sm" onClick={clearAll}>
                    Clear All
                  </Button>
                </div>
              </div>
              <CardDescription>Select the files and reports you want to export</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {filteredOptions.map((option) => {
                  const Icon = option.icon
                  const isSelected = selectedItems.includes(option.id)

                  return (
                    <div
                      key={option.id}
                      className={`flex items-center space-x-3 p-4 rounded-lg border cursor-pointer transition-all ${
                        isSelected ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
                      }`}
                      onClick={() => toggleItem(option.id)}
                    >
                      <Checkbox checked={isSelected} onChange={() => toggleItem(option.id)} />
                      <Icon className={`h-5 w-5 ${isSelected ? "text-primary" : "text-muted-foreground"}`} />
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-foreground">{option.name}</h4>
                          <div className="flex items-center space-x-2">
                            <Badge variant="outline">{option.format}</Badge>
                            <span className="text-sm text-muted-foreground">{option.size}</span>
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{option.description}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Export Summary & Controls */}
        <div className="space-y-4">
          <Card className="gradient-bg">
            <CardHeader>
              <CardTitle>Export Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Selected Items:</span>
                  <span className="font-medium">{selectedItems.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Total Size:</span>
                  <span className="font-medium">{getTotalSize()} MB</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Estimated Time:</span>
                  <span className="font-medium">{Math.ceil(selectedItems.length * 0.5)} min</span>
                </div>
              </div>

              <Button onClick={startExport} disabled={selectedItems.length === 0 || isExporting} className="w-full">
                <Download className="h-4 w-4 mr-2" />
                {isExporting ? "Exporting..." : "Start Export"}
              </Button>

              {isExporting && (
                <div className="space-y-2">
                  <Progress value={exportProgress} className="w-full" />
                  <p className="text-sm text-muted-foreground text-center">Preparing files... {exportProgress}%</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Export Guidelines */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Export Guidelines</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Academic Use</p>
                  <p className="text-xs text-muted-foreground">Reports formatted for dissertation submission</p>
                </div>
              </div>

              <div className="flex items-start space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Business Implementation</p>
                  <p className="text-xs text-muted-foreground">Practical guides for supermarket deployment</p>
                </div>
              </div>

              <div className="flex items-start space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Data Privacy</p>
                  <p className="text-xs text-muted-foreground">All sensitive data anonymized and aggregated</p>
                </div>
              </div>

              <div className="flex items-start space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Reproducibility</p>
                  <p className="text-xs text-muted-foreground">Includes code and parameters for replication</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Export Error */}
      {exportError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Export Failed:</strong> {exportError}
            <br />
            <Button 
              variant="outline" 
              size="sm" 
              className="mt-2"
              onClick={() => {
                setExportError(null)
                startExport()
              }}
            >
              Try Again
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Export Complete */}
      {exportComplete && exportResults && (
        <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800 dark:text-green-200">
            <div className="space-y-2">
              <p className="font-medium">Export completed successfully!</p>
              <p>{exportResults.files.length} files have been generated:</p>
              <ul className="list-disc list-inside text-sm">
                {exportResults.files.map((file, index) => (
                  <li key={index}>{file.split('/').pop()}</li>
                ))}
              </ul>
              <p className="text-sm">Files are saved in the exports directory and ready for download.</p>
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* Export History */}
      {exportComplete && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Exports</CardTitle>
            <CardDescription>Your export history and download links</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-muted/20 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Package className="h-5 w-5 text-primary" />
                  <div>
                    <p className="font-medium">Dynamic Pricing Analysis - Complete Package</p>
                    <p className="text-sm text-muted-foreground">
                      Exported {new Date().toLocaleString()} • {getTotalSize()} MB
                    </p>
                  </div>
                </div>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
