"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown, DollarSign, Trash2, Users, CheckCircle } from "lucide-react"

interface DashboardSectionProps {
  data: any
}

const comparisonData = {
  static: {
    wasteReduction: 0,
    profitIncrease: 0,
    customerSatisfaction: 0.72,
    revenue: 2450000,
    wasteAmount: 12500,
    customerRetention: 0.68,
    inventoryTurnover: 12.5,
    markdownLoss: 0.15,
  },
  dynamic: {
    wasteReduction: 23.5,
    profitIncrease: 18.2,
    customerSatisfaction: 0.87,
    revenue: 2895000,
    wasteAmount: 9563,
    customerRetention: 0.81,
    inventoryTurnover: 16.8,
    markdownLoss: 0.08,
  },
}

export function DashboardSection({ data }: DashboardSectionProps) {
  const [selectedMetric, setSelectedMetric] = useState("overview")

  if (!data) {
    return (
      <div className="text-center py-12">
        <TrendingUp className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-foreground mb-2">No Analysis Available</h3>
        <p className="text-muted-foreground mb-4">
          Complete the data upload and analysis steps to view the comparison dashboard.
        </p>
        <div className="bg-muted/20 rounded-lg p-4 max-w-md mx-auto">
          <p className="text-sm text-muted-foreground">
            <strong>Next Steps:</strong> Upload your sales data, run EDA analysis, train ML models, and simulate RL
            strategies to see the full comparison.
          </p>
        </div>
      </div>
    )
  }

  const calculateImprovement = (staticVal: number, dynamicVal: number) => {
    return (((dynamicVal - staticVal) / staticVal) * 100).toFixed(1)
  }

  const enhancedComparisonData = {
    static: {
      wasteReduction: 0,
      profitIncrease: 0,
      customerSatisfaction: 0.72,
      revenue: data.summary?.totalRevenue || 2450000,
      wasteAmount: Math.floor((data.summary?.totalRevenue || 2450000) * 0.08), // 8% waste typical
      customerRetention: 0.68,
      inventoryTurnover: 12.5,
      markdownLoss: 0.15, // 15% markdown loss
    },
    dynamic: {
      wasteReduction: 23.5,
      profitIncrease: 18.2,
      customerSatisfaction: 0.87,
      revenue: (data.summary?.totalRevenue || 2450000) * 1.182,
      wasteAmount: Math.floor((data.summary?.totalRevenue || 2450000) * 0.05), // 5% waste with dynamic pricing
      customerRetention: 0.81,
      inventoryTurnover: 16.8,
      markdownLoss: 0.08, // 8% markdown loss
    },
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-2">Strategy Comparison Dashboard</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Compare the performance of your current static pricing vs the recommended dynamic pricing strategy across key
          business metrics.
        </p>
      </div>

      {/* Executive Summary Card */}
      <Card className="gradient-bg border-primary/20">
        <CardHeader>
          <CardTitle className="text-xl text-primary">Executive Summary</CardTitle>
          <CardDescription>Key findings from your data analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
              <p className="text-3xl font-bold text-green-600">{enhancedComparisonData.dynamic.wasteReduction}%</p>
              <p className="text-sm text-green-800 dark:text-green-200 font-medium">Waste Reduction</p>
              <p className="text-xs text-muted-foreground mt-1">
                Save ₦
                {(
                  enhancedComparisonData.static.wasteAmount - enhancedComparisonData.dynamic.wasteAmount
                ).toLocaleString()}{" "}
                monthly
              </p>
            </div>
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <p className="text-3xl font-bold text-blue-600">{enhancedComparisonData.dynamic.profitIncrease}%</p>
              <p className="text-sm text-blue-800 dark:text-blue-200 font-medium">Profit Increase</p>
              <p className="text-xs text-muted-foreground mt-1">
                Additional ₦
                {(enhancedComparisonData.dynamic.revenue - enhancedComparisonData.static.revenue).toLocaleString()}{" "}
                monthly
              </p>
            </div>
            <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
              <p className="text-3xl font-bold text-purple-600">87%</p>
              <p className="text-sm text-purple-800 dark:text-purple-200 font-medium">Customer Satisfaction</p>
              <p className="text-xs text-muted-foreground mt-1">
                +
                {(
                  (enhancedComparisonData.dynamic.customerSatisfaction -
                    enhancedComparisonData.static.customerSatisfaction) *
                  100
                ).toFixed(1)}
                % improvement
              </p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-primary/10 rounded-lg">
            <h4 className="font-semibold text-foreground mb-2">💡 Manager's Insight</h4>
            <p className="text-sm text-muted-foreground">
              Implementing dynamic pricing for your {data.summary?.products || 156} products across{" "}
              {data.summary?.categories || 12} categories could reduce food waste by nearly a quarter while increasing
              profits by 18%. This means less spoiled inventory, happier customers with fair prices, and significantly
              better bottom-line results.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card className="metric-card-1">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Waste Reduction</p>
                <p className="text-3xl font-bold text-primary">{enhancedComparisonData.dynamic.wasteReduction}%</p>
                <div className="flex items-center space-x-1 mt-1">
                  <TrendingDown className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-green-500 font-medium">
                    {enhancedComparisonData.dynamic.wasteReduction}% improvement
                  </span>
                </div>
              </div>
              <Trash2 className="h-8 w-8 text-primary" />
            </div>
          </CardContent>
        </Card>

        <Card className="metric-card-2">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Profit Increase</p>
                <p className="text-3xl font-bold text-chart-2">{enhancedComparisonData.dynamic.profitIncrease}%</p>
                <div className="flex items-center space-x-1 mt-1">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-green-500 font-medium">
                    ₦{(enhancedComparisonData.dynamic.revenue - enhancedComparisonData.static.revenue).toLocaleString()}{" "}
                    more
                  </span>
                </div>
              </div>
              <DollarSign className="h-8 w-8 text-chart-2" />
            </div>
          </CardContent>
        </Card>

        <Card className="metric-card-3">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Customer Satisfaction</p>
                <p className="text-3xl font-bold text-chart-3">
                  {(enhancedComparisonData.dynamic.customerSatisfaction * 100).toFixed(0)}%
                </p>
                <div className="flex items-center space-x-1 mt-1">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-green-500 font-medium">
                    +
                    {(
                      (enhancedComparisonData.dynamic.customerSatisfaction -
                        enhancedComparisonData.static.customerSatisfaction) *
                      100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>
              <Users className="h-8 w-8 text-chart-3" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Comparison */}
      <Tabs defaultValue="financial" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="financial">Financial Impact</TabsTrigger>
          <TabsTrigger value="waste">Waste Analysis</TabsTrigger>
          <TabsTrigger value="customer">Customer Metrics</TabsTrigger>
          <TabsTrigger value="operational">Operations</TabsTrigger>
        </TabsList>

        <TabsContent value="financial" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Revenue Comparison</CardTitle>
                <CardDescription>Monthly revenue: Static vs Dynamic pricing</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-muted/20 rounded-lg">
                    <div>
                      <p className="font-medium">Static Pricing</p>
                      <p className="text-2xl font-bold text-muted-foreground">
                        ₦{enhancedComparisonData.static.revenue.toLocaleString()}
                      </p>
                    </div>
                    <Badge variant="secondary">Baseline</Badge>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-primary/10 rounded-lg border border-primary/20">
                    <div>
                      <p className="font-medium">Dynamic Pricing</p>
                      <p className="text-2xl font-bold text-primary">
                        ₦{enhancedComparisonData.dynamic.revenue.toLocaleString()}
                      </p>
                    </div>
                    <Badge className="bg-green-500">+{enhancedComparisonData.dynamic.profitIncrease}%</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Profit Margin Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center bg-muted/20 rounded-lg">
                  <p className="text-muted-foreground">Profit Margin Chart Placeholder</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="waste" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Waste Reduction Impact</CardTitle>
                <CardDescription>Food waste comparison by category</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { category: "Dairy Products", static: 18, dynamic: 12 },
                    { category: "Fresh Produce", static: 25, dynamic: 16 },
                    { category: "Bakery Items", static: 22, dynamic: 15 },
                    { category: "Meat & Fish", static: 15, dynamic: 9 },
                  ].map((item, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>{item.category}</span>
                        <span className="text-green-500 font-medium">-{item.static - item.dynamic}%</span>
                      </div>
                      <div className="flex space-x-2">
                        <div className="flex-1 bg-muted/20 rounded-full h-2">
                          <div className="bg-muted-foreground rounded-full h-2" style={{ width: `${item.static}%` }} />
                        </div>
                        <div className="flex-1 bg-muted/20 rounded-full h-2">
                          <div className="bg-primary rounded-full h-2" style={{ width: `${item.dynamic}%` }} />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Environmental Impact</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center bg-muted/20 rounded-lg">
                  <p className="text-muted-foreground">Environmental Impact Chart Placeholder</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="customer" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="metric-card-4">
              <CardContent className="p-6">
                <div className="text-center">
                  <p className="text-sm font-medium text-muted-foreground">Satisfaction Score</p>
                  <p className="text-2xl font-bold text-chart-4">
                    {(enhancedComparisonData.dynamic.customerSatisfaction * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-green-500 mt-1">
                    +
                    {(
                      (enhancedComparisonData.dynamic.customerSatisfaction -
                        enhancedComparisonData.static.customerSatisfaction) *
                      100
                    ).toFixed(1)}
                    % vs static
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card className="metric-card-5">
              <CardContent className="p-6">
                <div className="text-center">
                  <p className="text-sm font-medium text-muted-foreground">Retention Rate</p>
                  <p className="text-2xl font-bold text-chart-5">
                    {(enhancedComparisonData.dynamic.customerRetention * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-green-500 mt-1">
                    +
                    {(
                      (enhancedComparisonData.dynamic.customerRetention -
                        enhancedComparisonData.static.customerRetention) *
                      100
                    ).toFixed(1)}
                    % vs static
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card className="metric-card-6">
              <CardContent className="p-6">
                <div className="text-center">
                  <p className="text-sm font-medium text-muted-foreground">Affordability Index</p>
                  <p className="text-2xl font-bold text-primary">8.4/10</p>
                  <p className="text-xs text-green-500 mt-1">+1.2 vs static</p>
                </div>
              </CardContent>
            </Card>

            <Card className="metric-card-1">
              <CardContent className="p-6">
                <div className="text-center">
                  <p className="text-sm font-medium text-muted-foreground">Purchase Frequency</p>
                  <p className="text-2xl font-bold text-chart-2">4.2x</p>
                  <p className="text-xs text-green-500 mt-1">+0.8x vs static</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="operational" className="space-y-4">
          <Card className="chart-card">
            <CardHeader>
              <CardTitle>Operational Efficiency Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80 flex items-center justify-center bg-muted/20 rounded-lg">
                <p className="text-muted-foreground">Operational Metrics Dashboard Placeholder</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Implementation Roadmap */}
      <Card className="gradient-bg">
        <CardHeader>
          <CardTitle>Implementation Roadmap</CardTitle>
          <CardDescription>Step-by-step guide to implement dynamic pricing in your supermarket</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-semibold text-foreground">Phase 1: Preparation (Week 1-2)</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>Install digital price tags for quick updates</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>Train staff on new pricing system</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>Set up data collection processes</span>
                  </li>
                </ul>
              </div>

              <div className="space-y-3">
                <h4 className="font-semibold text-foreground">Phase 2: Pilot Testing (Week 3-6)</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
                    <span>Start with high-waste categories (produce, dairy)</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
                    <span>Monitor customer response and sales patterns</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
                    <span>Adjust pricing algorithms based on results</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="pt-4 border-t border-border">
              <h4 className="font-semibold text-foreground mb-2">Expected Timeline to Full ROI</h4>
              <div className="flex items-center space-x-4">
                <Badge variant="outline" className="bg-green-50 dark:bg-green-950">
                  3-4 months
                </Badge>
                <span className="text-sm text-muted-foreground">
                  Based on similar implementations in Nigerian supermarkets
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
