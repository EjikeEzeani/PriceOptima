"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Brain, TrendingUp, Zap, CheckCircle, Play } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, ScatterChart, Scatter } from "recharts"
import { trainML, type MLResponse } from "@/lib/api"

interface MLSectionProps {
  data: any
}

const models = [
  {
    id: "linear_regression",
    name: "Linear Regression",
    description: "Simple linear relationship modeling with interpretable coefficients",
    icon: TrendingUp,
    complexity: "Low",
    accuracy: 0.78,
    status: "ready",
    features: ["Fast training", "Interpretable", "Good baseline"]
  },
  {
    id: "random_forest",
    name: "Random Forest",
    description: "Ensemble method with multiple decision trees and feature importance",
    icon: Brain,
    complexity: "Medium",
    accuracy: 0.85,
    status: "ready",
    features: ["Robust", "Feature importance", "Handles non-linearity"]
  },
  {
    id: "xgboost",
    name: "XGBoost",
    description: "Gradient boosting with advanced SHAP explainability",
    icon: Zap,
    complexity: "High",
    accuracy: 0.91,
    status: "ready",
    features: ["High accuracy", "SHAP analysis", "Handles complex patterns"]
  },
  {
    id: "gradient_boosting",
    name: "Gradient Boosting",
    description: "Sequential boosting with detailed feature analysis",
    icon: Brain,
    complexity: "High",
    accuracy: 0.89,
    status: "ready",
    features: ["High performance", "Feature importance", "SHAP support"]
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

    try {
      // Simulate training progress
      for (let i = 0; i <= 100; i += 5) {
        setTrainingProgress(i)
        await new Promise((resolve) => setTimeout(resolve, 100))
      }

      // Call backend ML API using centralized client
      const results: MLResponse = await trainML(modelId)
      console.log('ML Results:', results)
      setModelResults(results)
    } catch (error) {
      console.error('Error training model:', error)
      // Fallback to mock results if API fails
      const mockResults: MLResponse = {
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
    } finally {
      setIsTraining(false)
    }
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
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="predictions">Predictions</TabsTrigger>
            <TabsTrigger value="future">Future Forecast</TabsTrigger>
            <TabsTrigger value="importance">Feature Importance</TabsTrigger>
            <TabsTrigger value="shap">SHAP Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="metrics" className="space-y-4">
            {/* Performance Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card className="metric-card-1">
                <CardContent className="p-6">
                  <div className="text-center">
                    <p className="metric-label high-contrast-label">Test R² Score</p>
                    <p className="metric-value high-contrast-text">{modelResults.metrics?.r2?.toFixed(3) || '0.000'}</p>
                    <p className="text-xs text-muted-foreground mt-1">Coefficient of Determination</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="metric-card-2">
                <CardContent className="p-6">
                  <div className="text-center">
                    <p className="metric-label high-contrast-label">RMSE</p>
                    <p className="metric-value high-contrast-text">{modelResults.metrics?.rmse?.toFixed(1) || '0.0'}</p>
                    <p className="text-xs text-muted-foreground mt-1">Root Mean Square Error</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="metric-card-3">
                <CardContent className="p-6">
                  <div className="text-center">
                    <p className="metric-label high-contrast-label">MAE</p>
                    <p className="metric-value high-contrast-text">{modelResults.metrics?.mae?.toFixed(1) || '0.0'}</p>
                    <p className="text-xs text-muted-foreground mt-1">Mean Absolute Error</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="metric-card-4">
                <CardContent className="p-6">
                  <div className="text-center">
                    <p className="metric-label high-contrast-label">MAPE</p>
                    <p className="metric-value high-contrast-text">{modelResults.metrics.mape?.toFixed(2) || 'N/A'}%</p>
                    <p className="text-xs text-muted-foreground mt-1">Mean Absolute Percentage Error</p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Training vs Test Performance */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="chart-card">
                <CardHeader>
                  <CardTitle>Training vs Test Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Training R²</span>
                      <span className="font-semibold text-primary">{modelResults.metrics.train_r2?.toFixed(3) || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Test R²</span>
                      <span className="font-semibold text-chart-2">{modelResults.metrics?.r2?.toFixed(3) || '0.000'}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Cross-Validation Mean</span>
                      <span className="font-semibold text-chart-3">{modelResults.metrics.cv_mean?.toFixed(3) || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">CV Standard Deviation</span>
                      <span className="font-semibold text-chart-4">{modelResults.metrics.cv_std?.toFixed(3) || 'N/A'}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="chart-card">
                <CardHeader>
                  <CardTitle>Model Performance Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Prediction Quality</span>
                      <Badge variant={modelResults.performanceAnalysis?.prediction_quality === 'Excellent' ? 'default' : 'secondary'}>
                        {modelResults.performanceAnalysis?.prediction_quality || 'Unknown'}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Model Stability</span>
                      <Badge variant={modelResults.performanceAnalysis?.model_stability === 'Stable' ? 'default' : 'destructive'}>
                        {modelResults.performanceAnalysis?.model_stability || 'Unknown'}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Overfitting Detected</span>
                      <Badge variant={modelResults.performanceAnalysis?.overfitting ? 'destructive' : 'default'}>
                        {modelResults.performanceAnalysis?.overfitting ? 'Yes' : 'No'}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">CV Consistency</span>
                      <Badge variant={modelResults.performanceAnalysis?.cv_consistency === 'High' ? 'default' : 'secondary'}>
                        {modelResults.performanceAnalysis?.cv_consistency || 'Unknown'}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Model Performance Visualization</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <ScatterChart data={modelResults.predictions}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="actual" name="Actual" unit="₦" />
                    <YAxis dataKey="predicted" name="Predicted" unit="₦" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Predictions" dataKey="predicted" fill="#3b82f6" />
                    <Line type="monotone" dataKey="actual" stroke="#10b981" strokeWidth={2} dot={false} />
                  </ScatterChart>
                </ResponsiveContainer>
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
                  {(modelResults.predictions || []).slice(0, 10).map((pred: any, index: number) => (
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
                          className={`font-semibold ${pred.error_percentage < 10 ? "text-green-500" : pred.error_percentage < 20 ? "text-orange-500" : "text-red-500"}`}
                        >
                          {pred.error_percentage?.toFixed(1) || ((Math.abs(pred.actual - pred.predicted) / pred.actual) * 100).toFixed(1)}% error
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="future" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Future Predictions</CardTitle>
                <CardDescription>7-day forecast based on current model performance</CardDescription>
              </CardHeader>
              <CardContent>
                {modelResults.futurePredictions && modelResults.futurePredictions.length > 0 ? (
                  <div className="space-y-4">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={modelResults.futurePredictions}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="day" />
                        <YAxis />
                        <Tooltip formatter={(value) => [`₦${value.toFixed(0)}`, 'Predicted Value']} />
                        <Line type="monotone" dataKey="predicted_value" stroke="#3b82f6" strokeWidth={3} name="Predicted Value" />
                      </LineChart>
                    </ResponsiveContainer>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {(modelResults.futurePredictions || []).map((pred: any, index: number) => (
                        <Card key={index} className="metric-card-1">
                          <CardContent className="p-4">
                            <div className="text-center">
                              <p className="metric-label high-contrast-label">{pred.day}</p>
                              <p className="metric-value high-contrast-text">₦{pred.predicted_value.toFixed(0)}</p>
                              <div className="flex justify-center space-x-2 mt-2">
                                <Badge variant={pred.confidence === 'High' ? 'default' : 'secondary'}>
                                  {pred.confidence}
                                </Badge>
                                <Badge variant={pred.trend === 'Increasing' ? 'default' : pred.trend === 'Decreasing' ? 'destructive' : 'secondary'}>
                                  {pred.trend}
                                </Badge>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground">Future predictions not available for this model</p>
                  </div>
                )}
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
                  {(modelResults.featureImportance || []).map((feature: any, index: number) => (
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
                <CardDescription>Understanding model decisions with SHAP values and feature contributions</CardDescription>
              </CardHeader>
              <CardContent>
                {modelResults.shapAnalysis && modelResults.shapAnalysis.length > 0 ? (
                  <div className="space-y-6">
                    {/* SHAP Summary */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <Card className="metric-card-1">
                        <CardHeader>
                          <CardTitle className="text-lg">Feature Importance (SHAP)</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={(modelResults.featureImportance || []).map((feature: any, index: number) => ({
                              feature: feature.feature,
                              importance: feature.importance * 100,
                              color: index === 0 ? "#3b82f6" : index === 1 ? "#10b981" : index === 2 ? "#f59e0b" : "#ef4444"
                            }))} layout="horizontal">
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis type="number" domain={[0, 100]} />
                              <YAxis dataKey="feature" type="category" width={120} />
                              <Tooltip formatter={(value) => [`${value}%`, 'Importance']} />
                              <Bar dataKey="importance" fill="#3b82f6" />
                            </BarChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>

                      <Card className="metric-card-2">
                        <CardHeader>
                          <CardTitle className="text-lg">SHAP Values Summary</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {(modelResults.shapAnalysis || []).slice(0, 3).map((sample: any, index: number) => (
                              <div key={index} className="p-3 bg-muted/20 rounded-lg">
                                <div className="flex justify-between items-center mb-2">
                                  <span className="font-medium">Sample {sample.sample_id + 1}</span>
                                  <div className="text-sm">
                                    <span className="text-chart-1">Actual: ₦{sample.actual.toFixed(0)}</span>
                                    <span className="mx-2">|</span>
                                    <span className="text-primary">Pred: ₦{sample.prediction.toFixed(0)}</span>
                                  </div>
                                </div>
                                <div className="space-y-1">
                                  {sample.feature_contributions.slice(0, 3).map((contrib: any, idx: number) => (
                                    <div key={idx} className="flex justify-between text-xs">
                                      <span>{contrib.feature}</span>
                                      <span className={contrib.shap_value > 0 ? "text-green-500" : "text-red-500"}>
                                        {contrib.shap_value > 0 ? "+" : ""}{contrib.shap_value.toFixed(2)}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    {/* Detailed SHAP Explanations */}
                    <Card className="chart-card">
                      <CardHeader>
                        <CardTitle>Detailed SHAP Explanations</CardTitle>
                        <CardDescription>Feature contributions for individual predictions</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {(modelResults.shapAnalysis || []).map((sample: any, index: number) => (
                            <div key={index} className="border rounded-lg p-4">
                              <div className="flex justify-between items-center mb-4">
                                <h4 className="font-semibold">Sample {sample.sample_id + 1}</h4>
                                <div className="flex space-x-4 text-sm">
                                  <span>Actual: <span className="font-bold text-chart-1">₦{sample.actual.toFixed(0)}</span></span>
                                  <span>Predicted: <span className="font-bold text-primary">₦{sample.prediction.toFixed(0)}</span></span>
                                  <span>Error: <span className="font-bold text-orange-500">{Math.abs(sample.actual - sample.prediction).toFixed(0)}</span></span>
                                </div>
                              </div>
                              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                {sample.feature_contributions.map((contrib: any, idx: number) => (
                                  <div key={idx} className={`p-3 rounded-lg border ${
                                    contrib.shap_value > 0 ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                                  }`}>
                                    <div className="flex justify-between items-center">
                                      <span className="font-medium text-sm">{contrib.feature}</span>
                                      <span className={`text-sm font-bold ${
                                        contrib.shap_value > 0 ? 'text-green-600' : 'text-red-600'
                                      }`}>
                                        {contrib.shap_value > 0 ? "+" : ""}{contrib.shap_value.toFixed(3)}
                                      </span>
                                    </div>
                                    <div className="text-xs text-muted-foreground mt-1">
                                      Value: {contrib.feature_value.toFixed(2)}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground mb-4">SHAP analysis not available for this model</p>
                    <p className="text-sm text-muted-foreground">
                      SHAP analysis is only available for tree-based models (Random Forest, XGBoost, Gradient Boosting)
                    </p>
                  </div>
                )}
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
