import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, AlertTriangle, CheckCircle, Clock, Lightbulb } from "lucide-react"

const insights = [
  {
    id: 1,
    title: "High demand detected for Rice products",
    description: "RL agent recommends 8% price increase for premium rice varieties",
    type: "opportunity",
    time: "2 minutes ago",
    impact: "High",
  },
  {
    id: 2,
    title: "Expiry alert: Dairy products",
    description: "47 dairy items expiring in 2 days - aggressive pricing recommended",
    type: "alert",
    time: "15 minutes ago",
    impact: "Critical",
  },
  {
    id: 3,
    title: "Model retrained successfully",
    description: "Random Forest model updated with latest sales data - accuracy improved to 94.8%",
    type: "success",
    time: "1 hour ago",
    impact: "Medium",
  },
  {
    id: 4,
    title: "Seasonal trend identified",
    description: "Increased demand for beverages detected - adjusting pricing strategy",
    type: "info",
    time: "3 hours ago",
    impact: "Medium",
  },
]

const getIcon = (type: string) => {
  switch (type) {
    case "opportunity":
      return TrendingUp
    case "alert":
      return AlertTriangle
    case "success":
      return CheckCircle
    default:
      return Clock
  }
}

const getTypeColor = (type: string) => {
  switch (type) {
    case "opportunity":
      return "bg-blue-500/20 text-blue-400 border-blue-500/30"
    case "alert":
      return "bg-red-500/20 text-red-400 border-red-500/30"
    case "success":
      return "bg-green-500/20 text-green-400 border-green-500/30"
    default:
      return "bg-purple-500/20 text-purple-400 border-purple-500/30"
  }
}

const getCardBackground = (type: string) => {
  switch (type) {
    case "opportunity":
      return "bg-gradient-to-r from-blue-500/5 to-cyan-500/5 border-blue-500/20"
    case "alert":
      return "bg-gradient-to-r from-red-500/5 to-orange-500/5 border-red-500/20"
    case "success":
      return "bg-gradient-to-r from-green-500/5 to-emerald-500/5 border-green-500/20"
    default:
      return "bg-gradient-to-r from-purple-500/5 to-pink-500/5 border-purple-500/20"
  }
}

export function RecentInsights() {
  return (
    <Card className="insight-card transition-all duration-300 hover:shadow-lg">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-full bg-gradient-to-r from-purple-500 to-pink-500">
            <Lightbulb className="h-5 w-5 text-white" />
          </div>
          <div>
            <CardTitle className="text-foreground">Recent Insights</CardTitle>
            <CardDescription className="text-muted-foreground">
              Latest recommendations and alerts from your AI models
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {insights.map((insight) => {
            const Icon = getIcon(insight.type)
            return (
              <div
                key={insight.id}
                className={`flex items-start space-x-4 p-4 rounded-xl border transition-all duration-300 hover:scale-[1.02] hover:shadow-md ${getCardBackground(insight.type)}`}
              >
                <div className="flex-shrink-0">
                  <div
                    className={`p-2 rounded-full ${
                      insight.type === "opportunity"
                        ? "bg-blue-500/20"
                        : insight.type === "alert"
                          ? "bg-red-500/20"
                          : insight.type === "success"
                            ? "bg-green-500/20"
                            : "bg-purple-500/20"
                    }`}
                  >
                    <Icon
                      className={`w-4 h-4 ${
                        insight.type === "opportunity"
                          ? "text-blue-400"
                          : insight.type === "alert"
                            ? "text-red-400"
                            : insight.type === "success"
                              ? "text-green-400"
                              : "text-purple-400"
                      }`}
                    />
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-semibold text-foreground">{insight.title}</p>
                    <div className="flex items-center space-x-2">
                      <Badge className={`${getTypeColor(insight.type)} px-2 py-1 text-xs font-medium`}>
                        {insight.impact}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2 leading-relaxed">{insight.description}</p>
                  <p className="text-xs text-muted-foreground font-medium">{insight.time}</p>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
