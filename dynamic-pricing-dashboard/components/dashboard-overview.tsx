import { MetricsGrid } from "@/components/metrics-grid"
import { PricingChart } from "@/components/pricing-chart"
import { ModelPerformance } from "@/components/model-performance"
import { RecentInsights } from "@/components/recent-insights"
import { Sparkles, Brain, TrendingUp } from "lucide-react"

export function DashboardOverview() {
  return (
    <div className="space-y-6">
      <div className="gradient-bg rounded-xl p-8 border border-border pulse-glow">
        <div className="max-w-4xl">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 rounded-full bg-gradient-to-r from-blue-500 to-purple-600">
              <Brain className="h-8 w-8 text-yellow-200" />
            </div>
            <div className="p-3 rounded-full bg-gradient-to-r from-green-500 to-emerald-600">
              <TrendingUp className="h-8 w-8 text-yellow-200" />
            </div>
            <div className="p-3 rounded-full bg-gradient-to-r from-pink-500 to-rose-600">
              <Sparkles className="h-8 w-8 text-yellow-200" />
            </div>
          </div>
          <h2 className="text-4xl font-bold bg-gradient-to-r from-cyan-300 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-3">
            AI-Powered Dynamic Pricing System
          </h2>
          <p className="text-xl text-muted-foreground leading-relaxed">
            Reducing food waste in Nigerian supermarkets through intelligent pricing strategies powered by machine
            learning and reinforcement learning algorithms.
          </p>
          <div className="mt-6 flex gap-4">
            <div className="px-4 py-2 bg-gradient-to-r from-green-600/40 to-emerald-600/40 rounded-full border border-green-400/50">
              <span className="text-green-300 font-semibold">34.2% Waste Reduction</span>
            </div>
            <div className="px-4 py-2 bg-gradient-to-r from-blue-600/40 to-purple-600/40 rounded-full border border-blue-400/50">
              <span className="text-blue-300 font-semibold">94.8% Accuracy</span>
            </div>
          </div>
        </div>
      </div>

      <MetricsGrid />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PricingChart />
        <ModelPerformance />
      </div>

      <RecentInsights />
    </div>
  )
}
