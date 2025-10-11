"use client"

import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends React.Component<{ children: React.ReactNode }, ErrorBoundaryState> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    if (error.message && (error.message.includes("MetaMask") || error.message.includes("ethereum"))) {
      console.warn("Suppressed Web3 error in ErrorBoundary:", error.message)
      return { hasError: false }
    }
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    if (!error.message?.includes("MetaMask") && !error.message?.includes("ethereum")) {
      console.error("Dashboard error:", error, errorInfo)
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background flex items-center justify-center p-4">
          <Card className="w-full max-w-md">
            <CardHeader className="text-center">
              <div className="mx-auto mb-4 p-3 rounded-full bg-red-500/10">
                <AlertTriangle className="h-8 w-8 text-red-500" />
              </div>
              <CardTitle className="text-red-600">Something went wrong</CardTitle>
              <CardDescription>
                The dashboard encountered an unexpected error. Please try refreshing the page.
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <Button onClick={() => window.location.reload()} className="w-full" variant="outline">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh Page
              </Button>
            </CardContent>
          </Card>
        </div>
      )
    }

    return this.props.children
  }
}
