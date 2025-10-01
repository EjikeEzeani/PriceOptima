import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, DollarSign, Package, Target, Zap } from "lucide-react"

const metrics = [
  {
    title: "Waste Reduction",
    value: "34.2%",
    change: "+12.3%",
    trend: "up",
    icon: TrendingDown,
    description: "vs last month",
    cardClass: "metric-card-1",
    iconColor: "text-green-500",
    valueColor: "text-green-400",
  },
  {
    title: "Revenue Impact",
    value: "₦2.4M",
    change: "+18.7%",
    trend: "up",
    icon: DollarSign,
    description: "additional revenue",
    cardClass: "metric-card-2",
    iconColor: "text-blue-500",
    valueColor: "text-blue-400",
  },
  {
    title: "Products Optimized",
    value: "1,247",
    change: "+156",
    trend: "up",
    icon: Package,
    description: "active SKUs",
    cardClass: "metric-card-3",
    iconColor: "text-purple-500",
    valueColor: "text-purple-400",
  },
  {
    title: "Model Accuracy",
    value: "94.8%",
    change: "+2.1%",
    trend: "up",
    icon: Target,
    description: "prediction accuracy",
    cardClass: "metric-card-4",
    iconColor: "text-pink-500",
    valueColor: "text-pink-400",
  },
  {
    title: "RL Agent Score",
    value: "847.3",
    change: "+23.4",
    trend: "up",
    icon: Zap,
    description: "reward score",
    cardClass: "metric-card-5",
    iconColor: "text-orange-500",
    valueColor: "text-orange-400",
  },
  {
    title: "Price Updates",
    value: "2,341",
    change: "+341",
    trend: "up",
    icon: TrendingUp,
    description: "today",
    cardClass: "metric-card-6",
    iconColor: "text-cyan-500",
    valueColor: "text-cyan-400",
  },
]

export function MetricsGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {metrics.map((metric) => (
        <Card
          key={metric.title}
          className={`${metric.cardClass} transition-all duration-300 hover:scale-105 hover:shadow-lg`}
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">{metric.title}</CardTitle>
            <div
              className={`p-2 rounded-full bg-gradient-to-r ${metric.iconColor.replace("text-", "from-")} ${metric.iconColor.replace("text-", "to-")}/80`}
            >
              <metric.icon className={`h-4 w-4 text-white`} />
            </div>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${metric.valueColor} mb-1`}>{metric.value}</div>
            <div className="flex items-center space-x-2 text-xs">
              <span
                className={`font-semibold px-2 py-1 rounded-full ${
                  metric.trend === "up"
                    ? "bg-green-500/20 text-green-400 border border-green-500/30"
                    : "bg-red-500/20 text-red-400 border border-red-500/30"
                }`}
              >
                {metric.change}
              </span>
              <span className="text-muted-foreground">{metric.description}</span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
