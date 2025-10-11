"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Brain, Cpu, Activity, Clock } from "lucide-react"

const models = [
  {
    name: "Random Forest",
    accuracy: 94.8,
    status: "active",
    type: "Supervised ML",
    icon: Brain,
    color: "from-green-500 to-emerald-600",
    progressColor: "bg-green-500",
  },
  {
    name: "XGBoost",
    accuracy: 92.3,
    status: "active",
    type: "Supervised ML",
    icon: Cpu,
    color: "from-blue-500 to-cyan-600",
    progressColor: "bg-blue-500",
  },
  {
    name: "Deep Q-Network",
    accuracy: 89.7,
    status: "training",
    type: "Reinforcement Learning",
    icon: Activity,
    color: "from-purple-500 to-pink-600",
    progressColor: "bg-purple-500",
  },
  {
    name: "LSTM Predictor",
    accuracy: 87.2,
    status: "active",
    type: "Time Series",
    icon: Clock,
    color: "from-orange-500 to-red-600",
    progressColor: "bg-orange-500",
  },
]

export function ModelPerformance() {
  return (
    <Card className="chart-card transition-all duration-300 hover:scale-[1.02] hover:shadow-xl">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-full bg-gradient-to-r from-purple-500 to-pink-500">
            <Brain className="h-5 w-5 text-white" />
          </div>
          <div>
            <CardTitle className="text-foreground">Model Performance</CardTitle>
            <CardDescription className="text-muted-foreground">
              Current accuracy scores across all models
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {models.map((model) => {
          const Icon = model.icon
          return (
            <div
              key={model.name}
              className="space-y-3 p-4 rounded-lg bg-gradient-to-r from-slate-800/50 to-slate-700/30 border border-slate-600/30 transition-all duration-300 hover:scale-[1.02]"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-full bg-gradient-to-r ${model.color}`}>
                    <Icon className="h-4 w-4 text-white" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-semibold text-foreground">{model.name}</p>
                    <p className="text-xs text-muted-foreground">{model.type}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <Badge
                    variant={model.status === "active" ? "default" : "secondary"}
                    className={
                      model.status === "active"
                        ? "bg-green-500/20 text-green-400 border-green-500/30 px-3 py-1"
                        : "bg-yellow-500/20 text-yellow-400 border-yellow-500/30 px-3 py-1"
                    }
                  >
                    {model.status}
                  </Badge>
                  <span className="text-lg font-bold text-foreground">{model.accuracy}%</span>
                </div>
              </div>
              <div className="space-y-2">
                <Progress value={model.accuracy} className="h-3" />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Performance</span>
                  <span
                    className={`font-semibold ${
                      model.accuracy >= 90
                        ? "text-green-400"
                        : model.accuracy >= 85
                          ? "text-yellow-400"
                          : "text-red-400"
                    }`}
                  >
                    {model.accuracy >= 90 ? "Excellent" : model.accuracy >= 85 ? "Good" : "Needs Improvement"}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}
