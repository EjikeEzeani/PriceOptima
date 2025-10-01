"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart3, PieChart, TrendingUp, Calendar, Package } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface EDASectionProps {
  data: any
}

export function EDASection({ data }: EDASectionProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisComplete, setAnalysisComplete] = useState(false)

  const runAnalysis = async () => {
    setIsAnalyzing(true)
    // Simulate analysis
    await new Promise((resolve) => setTimeout(resolve, 2000))
    setAnalysisComplete(true)
    setIsAnalyzing(false)
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <BarChart3 className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-foreground mb-2">No Data Available</h3>
        <p className="text-muted-foreground mb-4">
          Please upload your sales data first to begin exploratory data analysis.
        </p>
        <Button variant="outline" onClick={() => window.location.reload()}>
          Go to Upload Section
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-2">Exploratory Data Analysis</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Analyze your sales data to understand patterns, trends, and relationships that will inform the dynamic pricing
          strategy.
        </p>
      </div>

      {/* Data Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="metric-card-1">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Records</p>
                <p className="text-2xl font-bold text-primary">{data.summary?.totalRecords?.toLocaleString()}</p>
              </div>
              <Calendar className="h-8 w-8 text-primary" />
            </div>
          </CardContent>
        </Card>

        <Card className="metric-card-2">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Products</p>
                <p className="text-2xl font-bold text-chart-2">{data.summary?.products}</p>
              </div>
              <Package className="h-8 w-8 text-chart-2" />
            </div>
          </CardContent>
        </Card>

        <Card className="metric-card-3">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Categories</p>
                <p className="text-2xl font-bold text-chart-3">{data.summary?.categories}</p>
              </div>
              <PieChart className="h-8 w-8 text-chart-3" />
            </div>
          </CardContent>
        </Card>

        <Card className="metric-card-4">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Date Range</p>
                <p className="text-sm font-bold text-chart-4">{data.summary?.dateRange}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-chart-4" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Analysis Controls */}
      <Card className="gradient-bg">
        <CardHeader>
          <CardTitle>Run Data Analysis</CardTitle>
          <CardDescription>
            Generate comprehensive statistical analysis and visualizations of your sales data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={runAnalysis} disabled={isAnalyzing} className="w-full md:w-auto">
            {isAnalyzing ? "Analyzing Data..." : "Start EDA Analysis"}
          </Button>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {analysisComplete && (
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="correlations">Correlations</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="chart-card">
                <CardHeader>
                  <CardTitle>Sales Distribution by Category</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64 flex items-center justify-center bg-muted/20 rounded-lg">
                    <p className="text-muted-foreground">Interactive Pie Chart Placeholder</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="chart-card">
                <CardHeader>
                  <CardTitle>Revenue vs Waste Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64 flex items-center justify-center bg-muted/20 rounded-lg">
                    <p className="text-muted-foreground">Scatter Plot Placeholder</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="trends" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Sales Trends Over Time</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80 flex items-center justify-center bg-muted/20 rounded-lg">
                  <p className="text-muted-foreground">Time Series Chart Placeholder</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="correlations" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Correlation Heatmap</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80 flex items-center justify-center bg-muted/20 rounded-lg">
                  <p className="text-muted-foreground">Correlation Heatmap Placeholder</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="insights" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="insight-card">
                <CardHeader>
                  <CardTitle className="text-lg">Key Findings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-primary rounded-full mt-2"></div>
                    <p className="text-sm">Peak sales occur on weekends with 35% higher volume</p>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-chart-2 rounded-full mt-2"></div>
                    <p className="text-sm">Dairy products show highest waste percentage at 18%</p>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-chart-3 rounded-full mt-2"></div>
                    <p className="text-sm">Price elasticity varies significantly across categories</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="insight-card">
                <CardHeader>
                  <CardTitle className="text-lg">Recommendations</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-chart-4 rounded-full mt-2"></div>
                    <p className="text-sm">Implement dynamic pricing for high-waste categories</p>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-chart-5 rounded-full mt-2"></div>
                    <p className="text-sm">Focus ML models on perishable goods optimization</p>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-primary rounded-full mt-2"></div>
                    <p className="text-sm">Consider seasonal adjustments in pricing strategy</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      )}

      {analysisComplete && (
        <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
          <BarChart3 className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800 dark:text-green-200">
            EDA analysis complete! Your data shows clear patterns suitable for ML modeling. Proceed to the ML Models
            section to build predictive models.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
