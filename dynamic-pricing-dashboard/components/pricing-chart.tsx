"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts"
import { TrendingUp, BarChart3 } from "lucide-react"

const data = [
  { time: "00:00", baseline: 100, optimized: 98, demand: 45 },
  { time: "04:00", baseline: 100, optimized: 102, demand: 32 },
  { time: "08:00", baseline: 100, optimized: 105, demand: 78 },
  { time: "12:00", baseline: 100, optimized: 108, demand: 95 },
  { time: "16:00", baseline: 100, optimized: 103, demand: 87 },
  { time: "20:00", baseline: 100, optimized: 96, demand: 65 },
  { time: "24:00", baseline: 100, optimized: 94, demand: 42 },
]

export function PricingChart() {
  return (
    <Card className="chart-card transition-all duration-300 hover:scale-[1.02] hover:shadow-xl">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500">
            <BarChart3 className="h-5 w-5 text-white" />
          </div>
          <div>
            <CardTitle className="text-foreground flex items-center gap-2">
              Dynamic Pricing Performance
              <TrendingUp className="h-4 w-4 text-green-400" />
            </CardTitle>
            <CardDescription className="text-muted-foreground">
              Real-time price optimization vs baseline pricing
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="optimized" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="rgb(59 130 246)" stopOpacity={0.4} />
                <stop offset="50%" stopColor="rgb(147 51 234)" stopOpacity={0.2} />
                <stop offset="95%" stopColor="rgb(236 72 153)" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="baseline" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="rgb(156 163 175)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="rgb(75 85 99)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgb(59 130 246 / 0.2)" />
            <XAxis dataKey="time" stroke="rgb(161 161 170)" fontSize={12} />
            <YAxis stroke="rgb(161 161 170)" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgb(15 15 20 / 0.95)",
                border: "1px solid rgb(59 130 246 / 0.3)",
                borderRadius: "12px",
                color: "rgb(250 250 250)",
                backdropFilter: "blur(10px)",
                boxShadow: "0 8px 32px rgb(0 0 0 / 0.3)",
              }}
            />
            <Area
              type="monotone"
              dataKey="baseline"
              stroke="rgb(156 163 175)"
              fillOpacity={1}
              fill="url(#baseline)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="optimized"
              stroke="rgb(59 130 246)"
              fillOpacity={1}
              fill="url(#optimized)"
              strokeWidth={3}
            />
          </AreaChart>
        </ResponsiveContainer>
        <div className="flex justify-between mt-4 pt-4 border-t border-border">
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Avg Optimization</div>
            <div className="text-lg font-bold text-blue-400">+3.2%</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Peak Performance</div>
            <div className="text-lg font-bold text-green-400">+8.0%</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Efficiency Score</div>
            <div className="text-lg font-bold text-purple-400">92.4</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
