"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Brain, TrendingUp, Zap, CheckCircle, Play } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface MLSectionProps {
  data: any
}

const models = [
  {
    id: "linear",
    name: "Linear Regression",
    description: "Simple linear relationship modeling",
    icon: TrendingUp,
    complexity: "Low",
    accuracy: 0.78,
    status: "ready",
  },
  {
    id: "rf",
    name: "Random Forest",
    description: "Ensemble method with multiple decision trees",
    icon: Brain,
    complexity: "Medium",
    accuracy: 0.85,
    status: "ready",
  },
  {
    id: "xgboost",
    name: "XGBoost",
    description: "Gradient boosting with SHAP explainability",
    icon: Zap,
    complexity: "High",
    accuracy: 0.91,
    status: "ready",
  },
]

export function MLSection({ data }: MLSectionProps) {
  const [selectedModel, setSelectedModel] = useState("xgboost")
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [modelResults, setModelResults] = useState<any>(null)

  const trainModel = async (modelId: string) => {
    setIsTraining(true)
    setTrainingProgress(0)

    // Simulate training progress
    for (let i = 0; i <= 100; i += 5) {
      setTrainingProgress(i)
      await new Promise((resolve) => setTimeout(resolve, 100))
    }

    // Mock results
    const mockResults = {
      modelId,
      metrics: {
        r2: modelId === "xgboost" ? 0.91 : modelId === "rf" ? 0.85 : 0.78,
        rmse: modelId === "xgboost" ? 245.3 : modelId === "rf" ? 298.7 : 356.2,
        mae: modelId === "xgboost" ? 189.4 : modelId === "rf" ? 234.1 : 287.9,
      },
      predictions: Array.from({ length: 10 }, (_, i) => ({
        actual: 1000 + Math.random() * 500,
        predicted: 1000 + Math.random() * 500,
        product: `Product ${i + 1}`,
      })),
      featureImportance: [
        { feature: "Historical Sales", importance: 0.35 },
        { feature: "Seasonality", importance: 0.28 },
        { feature: "Price", importance: 0.22 },
        { feature: "Day of Week", importance: 0.15 },
      ],
    }

    setModelResults(mockResults)
    setIsTraining(false)
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <Brain className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-foreground mb-2">No Data Available</h3>
        <p className="text-muted-foreground mb-4">
          Please upload and analyze your data first before training ML models.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-2">Machine Learning Models</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Train and evaluate different ML models to predict optimal pricing and demand patterns for your supermarket
          products.
        </p>
      </div>

      {/* Model Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((model) => {
          const Icon = model.icon
          const isSelected = selectedModel === model.id

          return (
            <Card
              key={model.id}
              className={`cursor-pointer transition-all duration-200 ${
                isSelected ? "ring-2 ring-primary gradient-bg" : "hover:shadow-lg"
              }`}
              onClick={() => setSelectedModel(model.id)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <Icon className={`h-6 w-6 ${isSelected ? "text-primary" : "text-muted-foreground"}`} />
                  <Badge
                    variant={
                      model.complexity === "Low"
                        ? "secondary"
                        : model.complexity === "Medium"
                          ? "default"
                          : "destructive"
                    }
                  >
                    {model.complexity}
                  </Badge>
                </div>
                <CardTitle className="text-lg">{model.name}</CardTitle>
                <CardDescription className="text-sm">{model.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Expected Accuracy</span>
                  <span className="font-semibold text-primary">{(model.accuracy * 100).toFixed(1)}%</span>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Training Controls */}
      <Card className="gradient-bg">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Play className="h-5 w-5 text-primary" />
            <span>Train Selected Model</span>
          </CardTitle>
          <CardDescription>
            Train the {models.find((m) => m.id === selectedModel)?.name} model on your sales data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            <Button
              onClick={() => trainModel(selectedModel)}
              disabled={isTraining}
              className="flex items-center space-x-2"
            >
              <Brain className="h-4 w-4" />
              <span>{isTraining ? "Training..." : "Start Training"}</span>
            </Button>

            {isTraining && (
              <div className="flex-1 max-w-md">
                <Progress value={trainingProgress} className="w-full" />
                <p className="text-sm text-muted-foreground mt-1">Training progress: {trainingProgress}%</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {modelResults && (
        <Tabs defaultValue="metrics" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="predictions">Predictions</TabsTrigger>
            <TabsTrigger value="importance">Feature Importance</TabsTrigger>
            <TabsTrigger value="shap">SHAP Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="metrics" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="metric-card-1">
                <CardContent className="p-6">
                  <div className="text-center">
                    <p className="text-sm font-medium text-muted-foreground">R² Score</p>
                    <p className="text-3xl font-bold text-primary">{modelResults.metrics.r2.toFixed(3)}</p>
                    <p className="text-xs text-muted-foreground mt-1">Coefficient of Determination</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="metric-card-2">
                <CardContent className="p-6">
                  <div className="text-center">
                    <p className="text-sm font-medium text-muted-foreground">RMSE</p>
                    <p className="text-3xl font-bold text-chart-2">{modelResults.metrics.rmse.toFixed(1)}</p>
                    <p className="text-xs text-muted-foreground mt-1">Root Mean Square Error</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="metric-card-3">
                <CardContent className="p-6">
                  <div className="text-center">
                    <p className="text-sm font-medium text-muted-foreground">MAE</p>
                    <p className="text-3xl font-bold text-chart-3">{modelResults.metrics.mae.toFixed(1)}</p>
                    <p className="text-xs text-muted-foreground mt-1">Mean Absolute Error</p>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Model Performance Visualization</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center bg-muted/20 rounded-lg">
                  <p className="text-muted-foreground">Actual vs Predicted Chart Placeholder</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="predictions" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Sample Predictions</CardTitle>
                <CardDescription>Model predictions vs actual values for recent data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {modelResults.predictions.slice(0, 5).map((pred: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg">
                      <span className="font-medium">{pred.product}</span>
                      <div className="flex items-center space-x-4 text-sm">
                        <span>
                          Actual: <span className="font-semibold text-chart-1">₦{pred.actual.toFixed(0)}</span>
                        </span>
                        <span>
                          Predicted: <span className="font-semibold text-primary">₦{pred.predicted.toFixed(0)}</span>
                        </span>
                        <span
                          className={`font-semibold ${Math.abs(pred.actual - pred.predicted) / pred.actual < 0.1 ? "text-green-500" : "text-orange-500"}`}
                        >
                          {((Math.abs(pred.actual - pred.predicted) / pred.actual) * 100).toFixed(1)}% error
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="importance" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Feature Importance</CardTitle>
                <CardDescription>Which factors most influence pricing predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {modelResults.featureImportance.map((feature: any, index: number) => (
                    <div key={index} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{feature.feature}</span>
                        <span className="text-sm text-muted-foreground">{(feature.importance * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={feature.importance * 100} className="h-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="shap" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>SHAP Explainability Analysis</CardTitle>
                <CardDescription>Understanding model decisions with SHAP values</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80 flex items-center justify-center bg-muted/20 rounded-lg">
                  <p className="text-muted-foreground">SHAP Waterfall Plot Placeholder</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      {modelResults && (
        <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800 dark:text-green-200">
            Model training complete! R² score of {modelResults.metrics.r2.toFixed(3)} indicates good predictive
            performance. Proceed to RL Simulation to optimize pricing strategies.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
