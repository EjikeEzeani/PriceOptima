import { cn } from "@/lib/utils"
import { BarChart3, Brain, Database, TrendingUp, Settings, Target, Activity, PieChart } from "lucide-react"

const navigation = [
  { name: "Overview", icon: BarChart3, current: true },
  { name: "Data Analysis", icon: Database, current: false },
  { name: "ML Models", icon: Brain, current: false },
  { name: "RL Training", icon: Target, current: false },
  { name: "Pricing Engine", icon: TrendingUp, current: false },
  { name: "Performance", icon: Activity, current: false },
  { name: "Insights", icon: PieChart, current: false },
  { name: "Settings", icon: Settings, current: false },
]

export function Sidebar() {
  return (
    <div className="w-64 bg-card border-r border-border flex flex-col">
      <div className="p-6">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-foreground">PriceOptima</h1>
            <p className="text-xs text-muted-foreground">Dynamic Pricing AI</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-4 pb-4 space-y-1">
        {navigation.map((item) => (
          <a
            key={item.name}
            href="#"
            className={cn(
              "group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors",
              item.current
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground hover:bg-accent",
            )}
          >
            <item.icon className="mr-3 h-4 w-4 flex-shrink-0" />
            {item.name}
          </a>
        ))}
      </nav>

      <div className="p-4 border-t border-border">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
            <span className="text-sm font-medium">EE</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-foreground truncate">Ejike Ezeani</p>
            <p className="text-xs text-muted-foreground truncate">MSc Researcher</p>
          </div>
        </div>
      </div>
    </div>
  )
}
