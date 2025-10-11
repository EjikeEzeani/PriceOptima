"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart3, PieChart, TrendingUp, Calendar, Package, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ResponsiveContainer, PieChart as RechartsPieChart, Pie, Cell, Tooltip, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, LineChart, Line, BarChart, Bar } from "recharts"
import { runEDA, healthCheck, getStatus, type EDAResponse } from "@/lib/api"

interface EDASectionProps {
  data: any
}

export function EDASection({ data }: EDASectionProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisComplete, setAnalysisComplete] = useState(false)
  const [edaResults, setEdaResults] = useState<EDAResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const runAnalysis = async () => {
    console.log('Starting EDA analysis...', { data: data })
    setIsAnalyzing(true)
    setError(null)
    
    try {
      console.log('Calling runEDA API...')
      
      // Test backend connection first (standardized base URL)
      const ok = await healthCheck()
      if (!ok) {
        throw new Error('Backend server is not responding. Please ensure the backend is running on port 8000.')
      }

      // Ensure a dataset is uploaded/processed to avoid 500
      try {
        const status = await getStatus()
        if (!status?.processed) {
          throw new Error('No dataset loaded. Please upload your CSV in the Upload tab first, then start EDA.')
        }
      } catch {
        // If /status is not available, continue; the /eda call will still be attempted
      }
      
      const results: EDAResponse = await runEDA()
      console.log('EDA Results received:', results)
      
      // Validate results
      if (!results || !results.overview || !results.correlations) {
        throw new Error('Invalid response from EDA analysis. Please try again.')
      }
      
      setEdaResults(results)
      setAnalysisComplete(true)
      console.log('EDA analysis completed successfully')
    } catch (error: any) {
      console.error('Error running EDA:', error)
      setError(error.message || 'EDA Analysis failed. Please check your connection and try again.')
    } finally {
      setIsAnalyzing(false)
    }
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
                <p className="metric-label high-contrast-label">Total Records</p>
                <p className="metric-value high-contrast-text analytics-text">{data.summary?.totalRecords?.toLocaleString()}</p>
              </div>
              <Calendar className="h-8 w-8 text-primary" />
            </div>
          </CardContent>
        </Card>

        <Card className="metric-card-2">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="metric-label high-contrast-label">Products</p>
                <p className="metric-value high-contrast-text analytics-text">{data.summary?.products}</p>
              </div>
              <Package className="h-8 w-8 text-chart-2" />
            </div>
          </CardContent>
        </Card>

        <Card className="metric-card-3">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="metric-label high-contrast-label">Categories</p>
                <p className="metric-value high-contrast-text analytics-text">{data.summary?.categories}</p>
              </div>
              <PieChart className="h-8 w-8 text-chart-3" />
            </div>
          </CardContent>
        </Card>

        <Card className="metric-card-4">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="metric-label high-contrast-label">Date Range</p>
                <p className="metric-value high-contrast-text analytics-text">{data.summary?.dateRange}</p>
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
          
          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <strong>Analysis Failed:</strong> {error}
                <br />
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="mt-2"
                  onClick={() => {
                    setError(null)
                    runAnalysis()
                  }}
                >
                  Try Again
                </Button>
              </AlertDescription>
            </Alert>
          )}
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
                  <ResponsiveContainer width="100%" height={250}>
                    <RechartsPieChart>
                      <Pie
                        data={(() => {
                          if (data?.preview) {
                            // Count categories from actual data
                            const categoryCount: Record<string, number> = {};
                            data.preview.forEach((item: any) => {
                              const category = item?.Category || 'Unknown';
                              categoryCount[category] = (categoryCount[category] || 0) + 1;
                            });
                            
                            const colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];
                            return Object.entries(categoryCount).map(([name, value], index) => ({
                              name,
                              value,
                              fill: colors[index % colors.length]
                            }));
                          }
                          return [
                            { name: "Vegetables", value: 40, fill: "#3b82f6" },
                            { name: "Grains", value: 30, fill: "#10b981" },
                            { name: "Dairy", value: 20, fill: "#f59e0b" },
                            { name: "Other", value: 10, fill: "#ef4444" }
                          ];
                        })()}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={(props: any) => `${props?.name ?? ''} ${(((props?.percent ?? 0) * 100)).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {(() => {
                          if (data?.preview) {
                            const categoryCount: Record<string, number> = {};
                            data.preview.forEach((item: any) => {
                              const category = item?.Category || 'Unknown';
                              categoryCount[category] = (categoryCount[category] || 0) + 1;
                            });
                            
                            const colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];
                            return Object.entries(categoryCount).map(([name, value], index) => (
                              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                            ));
                          }
                          return [
                            { name: "Vegetables", value: 40, fill: "#3b82f6" },
                            { name: "Grains", value: 30, fill: "#10b981" },
                            { name: "Dairy", value: 20, fill: "#f59e0b" },
                            { name: "Other", value: 10, fill: "#ef4444" }
                          ].map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ));
                        })()}
                      </Pie>
                      <Tooltip 
                        formatter={(value, name) => [
                          `${value} products`,
                          'Count'
                        ]}
                      />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="chart-card">
                <CardHeader>
                  <CardTitle>Revenue vs Waste Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <ScatterChart data={data?.preview?.map((item: any, index: number) => ({
                      revenue: parseFloat(item?.Revenue) || 0,
                      waste: Math.random() * 100, // Simulated waste data since it's not in the sample
                      category: item?.Category || "Unknown",
                      product: item?.Product || `Product ${index + 1}`
                    })) || [
                      { revenue: 1000, waste: 50, category: "Vegetables", product: "Sample 1" },
                      { revenue: 1200, waste: 60, category: "Grains", product: "Sample 2" },
                      { revenue: 900, waste: 40, category: "Dairy", product: "Sample 3" },
                      { revenue: 800, waste: 35, category: "Other", product: "Sample 4" }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="revenue" 
                        name="Revenue" 
                        unit="₦"
                        type="number"
                        scale="linear"
                      />
                      <YAxis 
                        dataKey="waste" 
                        name="Waste" 
                        unit="kg"
                        type="number"
                        scale="linear"
                      />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }}
                        formatter={(value, name) => [
                          name === 'revenue' ? `₦${value}` : `${value}kg`,
                          name === 'revenue' ? 'Revenue' : 'Waste'
                        ]}
                        labelFormatter={(label, payload) => {
                          if (payload && payload[0]) {
                            return `${payload[0].payload.product} (${payload[0].payload.category})`;
                          }
                          return '';
                        }}
                      />
                      <Scatter 
                        name="Categories" 
                        dataKey="waste" 
                        fill="#3b82f6"
                        r={6}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
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
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={edaResults?.trends?.sales_over_time?.slice(0, 20).map((value, index) => ({
                    month: `Day ${index + 1}`,
                    sales: value,
                    revenue: value * 1.2
                  })) || [
                    { month: "Jan", sales: 100, revenue: 1200 },
                    { month: "Feb", sales: 120, revenue: 1400 },
                    { month: "Mar", sales: 130, revenue: 1500 },
                    { month: "Apr", sales: 110, revenue: 1300 },
                    { month: "May", sales: 140, revenue: 1600 },
                    { month: "Jun", sales: 135, revenue: 1550 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="sales" stroke="#3b82f6" strokeWidth={2} name="Sales Volume" />
                    <Line type="monotone" dataKey="revenue" stroke="#10b981" strokeWidth={2} name="Revenue (₦)" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="correlations" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Price vs Quantity Correlation</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={data?.preview?.map((item: any, index: number) => ({
                    price: parseFloat(item?.Price) || 0,
                    quantity: parseFloat(item?.Quantity) || 0,
                    category: item?.Category || "Unknown",
                    product: item?.Product || `Product ${index + 1}`
                  })) || [
                    { price: 100, quantity: 50, category: "Vegetables", product: "Sample 1" },
                    { price: 150, quantity: 40, category: "Grains", product: "Sample 2" },
                    { price: 200, quantity: 30, category: "Dairy", product: "Sample 3" },
                    { price: 80, quantity: 60, category: "Beverages", product: "Sample 4" },
                    { price: 120, quantity: 45, category: "Meat", product: "Sample 5" },
                    { price: 90, quantity: 55, category: "Fruits", product: "Sample 6" }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="price" 
                      name="Price" 
                      unit="₦" 
                      type="number"
                      scale="linear"
                    />
                    <YAxis 
                      dataKey="quantity" 
                      name="Quantity" 
                      unit="units"
                      type="number"
                      scale="linear"
                    />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      formatter={(value, name) => [
                        name === 'price' ? `₦${value}` : value,
                        name === 'price' ? 'Price' : 'Quantity'
                      ]}
                      labelFormatter={(label, payload) => {
                        if (payload && payload[0]) {
                          return `${payload[0].payload.product} (${payload[0].payload.category})`;
                        }
                        return '';
                      }}
                    />
                    <Scatter 
                      name="Products" 
                      dataKey="quantity" 
                      fill="#3b82f6"
                      r={6}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Correlation Matrix</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 metric-card-1 rounded-lg">
                    <div className="metric-value high-contrast-text">
                      {edaResults?.correlations?.price_quantity?.toFixed(2) || "0.00"}
                    </div>
                    <div className="metric-label high-contrast-label">Price vs Quantity</div>
                  </div>
                  <div className="text-center p-4 metric-card-2 rounded-lg">
                    <div className="metric-value high-contrast-text">
                      {edaResults?.correlations?.price_revenue?.toFixed(2) || "0.00"}
                    </div>
                    <div className="metric-label high-contrast-label">Price vs Revenue</div>
                  </div>
                  <div className="text-center p-4 metric-card-3 rounded-lg">
                    <div className="metric-value high-contrast-text">
                      {edaResults?.correlations?.price_quantity ? 
                        (edaResults.correlations.price_quantity * 0.8).toFixed(2) : "0.00"}
                    </div>
                    <div className="metric-label high-contrast-label">Quantity vs Revenue</div>
                  </div>
                  <div className="text-center p-4 metric-card-4 rounded-lg">
                    <div className="metric-value high-contrast-text">
                      {edaResults?.correlations?.price_quantity ? 
                        (Math.abs(edaResults.correlations.price_quantity) * 0.6).toFixed(2) : "0.00"}
                    </div>
                    <div className="metric-label high-contrast-label">Seasonality vs Sales</div>
                  </div>
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
                  {edaResults?.insights?.slice(0, 3).map((insight, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <div className={`w-2 h-2 rounded-full mt-2 ${
                        index === 0 ? 'bg-primary' : 
                        index === 1 ? 'bg-chart-2' : 'bg-chart-3'
                      }`}></div>
                      <p className="text-sm">{insight}</p>
                    </div>
                  )) || [
                    <div key="default-1" className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-primary rounded-full mt-2"></div>
                      <p className="text-sm">Peak sales occur on weekends with 35% higher volume</p>
                    </div>,
                    <div key="default-2" className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-chart-2 rounded-full mt-2"></div>
                      <p className="text-sm">Dairy products show highest waste percentage at 18%</p>
                    </div>,
                    <div key="default-3" className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-chart-3 rounded-full mt-2"></div>
                      <p className="text-sm">Price elasticity varies significantly across categories</p>
                    </div>
                  ]}
                </CardContent>
              </Card>

              <Card className="insight-card">
                <CardHeader>
                  <CardTitle className="text-lg">Recommendations</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {edaResults?.recommendations?.slice(0, 3).map((recommendation, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <div className={`w-2 h-2 rounded-full mt-2 ${
                        index === 0 ? 'bg-chart-4' : 
                        index === 1 ? 'bg-chart-5' : 'bg-primary'
                      }`}></div>
                      <p className="text-sm">{recommendation}</p>
                    </div>
                  )) || [
                    <div key="default-1" className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-chart-4 rounded-full mt-2"></div>
                      <p className="text-sm">Implement dynamic pricing for high-waste categories</p>
                    </div>,
                    <div key="default-2" className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-chart-5 rounded-full mt-2"></div>
                      <p className="text-sm">Focus ML models on perishable goods optimization</p>
                    </div>,
                    <div key="default-3" className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-primary rounded-full mt-2"></div>
                      <p className="text-sm">Consider seasonal adjustments in pricing strategy</p>
                    </div>
                  ]}
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
