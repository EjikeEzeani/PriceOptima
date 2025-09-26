import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Calendar, Download, RefreshCw } from "lucide-react"

export function Header() {
  return (
    <header className="bg-card border-b border-border px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Dynamic Pricing Dashboard</h1>
          <p className="text-muted-foreground">Waste Reduction for Nigerian Supermarkets</p>
        </div>

        <div className="flex items-center space-x-4">
          <Badge variant="secondary" className="bg-green-500/10 text-green-400 border-green-500/20">
            <div className="w-2 h-2 bg-green-400 rounded-full mr-2" />
            Models Active
          </Badge>

          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              className="border-yellow-500/50 bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20 hover:border-yellow-400"
            >
              <Calendar className="w-4 h-4 mr-2 text-yellow-400" />
              Last 7 days
            </Button>

            <Button
              variant="outline"
              size="sm"
              className="border-yellow-500/50 bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20 hover:border-yellow-400"
            >
              <RefreshCw className="w-4 h-4 mr-2 text-yellow-400" />
              Refresh
            </Button>

            <Button size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </div>
    </header>
  )
}
