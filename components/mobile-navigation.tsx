"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { 
  Upload, 
  BarChart3, 
  Brain, 
  Zap, 
  TrendingUp, 
  Download, 
  Shield,
  Menu,
  X
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

const tabs = [
  { id: "upload", label: "Upload", icon: Upload, description: "Upload data" },
  { id: "debug", label: "Debug", icon: Shield, description: "Test components" },
  { id: "eda", label: "Analysis", icon: BarChart3, description: "Data insights" },
  { id: "ml", label: "AI", icon: Brain, description: "ML predictions" },
  { id: "rl", label: "Pricing", icon: Zap, description: "Price optimization" },
  { id: "dashboard", label: "Results", icon: TrendingUp, description: "Compare strategies" },
  { id: "export", label: "Export", icon: Download, description: "Download reports" },
]

interface MobileNavigationProps {
  activeTab: string
  onTabChange: (tabId: string) => void
}

export function MobileNavigation({ activeTab, onTabChange }: MobileNavigationProps) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      {/* Mobile Navigation Toggle */}
      <div className="md:hidden fixed bottom-4 right-4 z-50">
        <Button
          onClick={() => setIsOpen(!isOpen)}
          className="w-14 h-14 rounded-full bg-primary text-primary-foreground shadow-lg hover:shadow-xl transition-all duration-200 touchable"
          size="lg"
        >
          {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </Button>
      </div>

      {/* Mobile Navigation Overlay */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="md:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Mobile Navigation Panel */}
      <motion.div
        initial={{ x: "100%" }}
        animate={{ x: isOpen ? 0 : "100%" }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", damping: 25, stiffness: 200 }}
        className="md:hidden fixed top-0 right-0 h-full w-80 bg-card border-l border-border z-50 overflow-y-auto"
      >
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-foreground">Navigation</h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsOpen(false)}
              className="p-2"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          <div className="space-y-2">
            {tabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id

              return (
                <Button
                  key={tab.id}
                  variant={isActive ? "default" : "ghost"}
                  onClick={() => {
                    onTabChange(tab.id)
                    setIsOpen(false)
                  }}
                  className={`w-full justify-start p-4 h-auto touchable ${
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-accent hover:text-accent-foreground"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <Icon className="h-5 w-5" />
                    <div className="text-left">
                      <div className="font-medium">{tab.label}</div>
                      <div className="text-sm opacity-80">{tab.description}</div>
                    </div>
                  </div>
                </Button>
              )
            })}
          </div>

          <div className="mt-8 pt-6 border-t border-border">
            <Card className="p-4 bg-muted/50">
              <h3 className="font-semibold text-sm text-foreground mb-2">Quick Tips</h3>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• Upload CSV files up to 50MB</li>
                <li>• Use the debug panel to test features</li>
                <li>• Export results in multiple formats</li>
                <li>• AI predictions update in real-time</li>
              </ul>
            </Card>
          </div>
        </div>
      </motion.div>

      {/* Bottom Navigation Bar for Mobile */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 bg-card border-t border-border z-30 mobile-nav-container">
        <div className="flex justify-around py-2">
          {tabs.slice(0, 5).map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id

            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex flex-col items-center p-2 rounded-lg transition-colors touchable ${
                  isActive
                    ? "text-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="h-5 w-5 mb-1" />
                <span className="text-xs font-medium">{tab.label}</span>
              </button>
            )
          })}
        </div>
      </div>
    </>
  )
}
