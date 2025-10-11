"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Users,
  ShoppingCart,
  Package,
  AlertTriangle,
  Shield,
  Lock,
  Eye,
  RefreshCw,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"

const generateRealtimeData = () => {
  const now = new Date()
  return Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    sales: Math.floor(Math.random() * 50000) + 20000,
    profit: Math.floor(Math.random() * 15000) + 5000,
    waste: Math.floor(Math.random() * 500) + 100,
  }))
}

const categoryData = [
  { category: "Fresh Produce", sales: 45000, profit: 12000, waste: 8 },
  { category: "Dairy", sales: 32000, profit: 9500, waste: 3 },
  { category: "Meat & Fish", sales: 28000, profit: 8400, waste: 12 },
  { category: "Bakery", sales: 22000, profit: 6600, waste: 15 },
  { category: "Beverages", sales: 35000, profit: 10500, waste: 2 },
]

export function AnalyticsDashboard() {
  const [realtimeData, setRealtimeData] = useState(generateRealtimeData())
  const [lastUpdated, setLastUpdated] = useState(new Date())
  const [isLive, setIsLive] = useState(true)

  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(() => {
      setRealtimeData(generateRealtimeData())
      setLastUpdated(new Date())
    }, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [isLive])

  const totalSales = realtimeData.reduce((sum, item) => sum + item.sales, 0)
  const totalProfit = realtimeData.reduce((sum, item) => sum + item.profit, 0)
  const totalWaste = realtimeData.reduce((sum, item) => sum + item.waste, 0)

  return (
    <div className="space-y-6">
      {/* Header with Live Status */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Real-Time Analytics</h2>
          <p className="text-muted-foreground">Live data from your stores</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isLive ? "bg-green-500 animate-pulse" : "bg-gray-400"}`} />
            <span className="text-sm text-muted-foreground">
              {isLive ? "Live" : "Paused"} • Updated {lastUpdated.toLocaleTimeString()}
            </span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsLive(!isLive)}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${isLive ? "animate-spin" : ""}`} />
            <span>{isLive ? "Pause" : "Resume"}</span>
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <Card className="metric-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Sales (24h)</CardTitle>
              <DollarSign className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">₦{totalSales.toLocaleString()}</div>
              <p className="text-xs text-green-600 flex items-center">
                <TrendingUp className="h-3 w-3 mr-1" />
                +12.5% from yesterday
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Card className="metric-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Profit Margin</CardTitle>
              <TrendingUp className="h-4 w-4 text-blue-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{((totalProfit / totalSales) * 100).toFixed(1)}%</div>
              <p className="text-xs text-blue-600 flex items-center">
                <TrendingUp className="h-3 w-3 mr-1" />
                +3.2% optimized
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="metric-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Waste Reduction</CardTitle>
              <Package className="h-4 w-4 text-orange-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{totalWaste}kg</div>
              <p className="text-xs text-orange-600 flex items-center">
                <TrendingDown className="h-3 w-3 mr-1" />
                -35% vs baseline
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Card className="metric-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Stores</CardTitle>
              <Users className="h-4 w-4 text-purple-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">127</div>
              <p className="text-xs text-purple-600 flex items-center">
                <Eye className="h-3 w-3 mr-1" />
                All systems online
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="chart-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Sales & Profit Trends (24h)</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={realtimeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="hour" stroke="hsl(var(--muted-foreground))" />
                <YAxis stroke="hsl(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Line type="monotone" dataKey="sales" stroke="#3b82f6" strokeWidth={2} name="Sales (₦)" />
                <Line type="monotone" dataKey="profit" stroke="#10b981" strokeWidth={2} name="Profit (₦)" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="chart-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <ShoppingCart className="h-5 w-5" />
              <span>Category Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={categoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="category" stroke="hsl(var(--muted-foreground))" />
                <YAxis stroke="hsl(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey="sales" fill="#3b82f6" name="Sales (₦)" />
                <Bar dataKey="profit" fill="#10b981" name="Profit (₦)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Security & Privacy Section */}
      <Card className="security-card border-green-200 bg-green-50/50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-green-800">
            <Shield className="h-5 w-5" />
            <span>Data Security & Privacy</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                <Lock className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <div className="font-semibold text-green-800">End-to-End Encryption</div>
                <div className="text-sm text-green-600">All data encrypted in transit and at rest</div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                <Eye className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <div className="font-semibold text-green-800">Privacy Compliant</div>
                <div className="text-sm text-green-600">GDPR & NDPR compliant data handling</div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                <Shield className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <div className="font-semibold text-green-800">Secure Infrastructure</div>
                <div className="text-sm text-green-600">ISO 27001 certified cloud hosting</div>
              </div>
            </div>
          </div>
          <div className="mt-4 p-3 bg-green-100 rounded-lg">
            <p className="text-sm text-green-800">
              <strong>Your data privacy is our priority.</strong> We never share customer data with third parties. All
              analytics are processed securely and anonymized for insights.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Alerts */}
      <Card className="alert-card border-orange-200 bg-orange-50/50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-orange-800">
            <AlertTriangle className="h-5 w-5" />
            <span>Smart Alerts</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-orange-100 rounded-lg">
              <div className="flex items-center space-x-3">
                <Package className="h-5 w-5 text-orange-600" />
                <div>
                  <div className="font-semibold text-orange-800">High Waste Alert - Dairy Section</div>
                  <div className="text-sm text-orange-600">Consider 15% price reduction on milk products</div>
                </div>
              </div>
              <Badge variant="outline" className="text-orange-600 border-orange-600">
                Action Required
              </Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-blue-100 rounded-lg">
              <div className="flex items-center space-x-3">
                <TrendingUp className="h-5 w-5 text-blue-600" />
                <div>
                  <div className="font-semibold text-blue-800">Opportunity Detected - Beverages</div>
                  <div className="text-sm text-blue-600">Increase prices by 8% based on demand surge</div>
                </div>
              </div>
              <Badge variant="outline" className="text-blue-600 border-blue-600">
                Opportunity
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
