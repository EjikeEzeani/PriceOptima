"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Zap, Target, TrendingUp, Brain, Play, Pause, RotateCcw, Settings } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar } from "recharts"

interface RLSectionProps {
  data: any
}

const algorithms = [
  {
    id: "qlearning",
    name: "Q-Learning",
    description: "Classic tabular reinforcement learning",
    complexity: "Medium",
    convergence: "Slow",
    stability: "High",
  },
  {
    id: "dqn",
    name: "Deep Q-Network (DQN)",
    description: "Neural network-based Q-learning",
    complexity: "High",
    convergence: "Fast",
    stability: "Medium",
  },
]

export function RLSection({ data }: RLSectionProps) {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("dqn")
  const [isTraining, setIsTraining] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [episode, setEpisode] = useState(0)
  const [totalEpisodes] = useState(1000)
  const [reward, setReward] = useState(0)
  const [results, setResults] = useState<any>(null)

  // Hyperparameters
  const [learningRate, setLearningRate] = useState([0.001])
  const [epsilon, setEpsilon] = useState([0.1])
  const [discountFactor, setDiscountFactor] = useState([0.95])

  const startTraining = async () => {
    setIsTraining(true)
    setIsPaused(false)
    setEpisode(0)
    setReward(0)

    // Simulate training
    for (let i = 0; i <= totalEpisodes; i += 10) {
      if (isPaused) break

      setEpisode(i)
      setReward((prev) => prev + Math.random() * 10 - 2) // Simulate learning
      await new Promise((resolve) => setTimeout(resolve, 50))
    }

    // Mock results
    const mockResults = {
      algorithm: selectedAlgorithm,
      finalReward: reward,
      convergenceEpisode: Math.floor(totalEpisodes * 0.7),
      policy: {
        wasteReduction: 23.5,
        profitIncrease: 18.2,
        customerSatisfaction: 0.87,
      },
      trainingCurve: Array.from({ length: 100 }, (_, i) => ({
        episode: i * 10,
        reward: Math.sin(i * 0.1) * 50 + i * 0.5 + Math.random() * 10,
      })),
    }

    setResults(mockResults)
    setIsTraining(false)
  }

  const pauseTraining = () => {
    setIsPaused(!isPaused)
  }

  const resetTraining = () => {
    setIsTraining(false)
    setIsPaused(false)
    setEpisode(0)
    setReward(0)
    setResults(null)
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <Zap className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-foreground mb-2">No Data Available</h3>
        <p className="text-muted-foreground mb-4">
          Please complete the ML modeling step before running RL simulations.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-2">Reinforcement Learning Simulation</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Train RL agents to learn optimal dynamic pricing policies that maximize profit while minimizing food waste
          through trial and error.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Algorithm Selection & Settings */}
        <div className="space-y-4">
          <Card className="gradient-bg">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="h-5 w-5 text-primary" />
                <span>Algorithm Selection</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {algorithms.map((algo) => (
                <div
                  key={algo.id}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedAlgorithm === algo.id
                      ? "border-primary bg-primary/10"
                      : "border-border hover:border-primary/50"
                  }`}
                  onClick={() => setSelectedAlgorithm(algo.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">{algo.name}</h4>
                    <Badge variant="outline">{algo.complexity}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">{algo.description}</p>
                  <div className="flex justify-between text-xs">
                    <span>Convergence: {algo.convergence}</span>
                    <span>Stability: {algo.stability}</span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Hyperparameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">Learning Rate: {learningRate[0]}</label>
                <Slider
                  value={learningRate}
                  onValueChange={setLearningRate}
                  max={0.01}
                  min={0.0001}
                  step={0.0001}
                  className="mt-2"
                />
              </div>

              <div>
                <label className="text-sm font-medium">Epsilon (Exploration): {epsilon[0]}</label>
                <Slider value={epsilon} onValueChange={setEpsilon} max={1} min={0.01} step={0.01} className="mt-2" />
              </div>

              <div>
                <label className="text-sm font-medium">Discount Factor: {discountFactor[0]}</label>
                <Slider
                  value={discountFactor}
                  onValueChange={setDiscountFactor}
                  max={0.99}
                  min={0.8}
                  step={0.01}
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Training Controls & Progress */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="gradient-bg">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="h-5 w-5 text-primary" />
                <span>Training Control</span>
              </CardTitle>
              <CardDescription>
                Train the {algorithms.find((a) => a.id === selectedAlgorithm)?.name} agent
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-4 mb-4">
                <Button onClick={startTraining} disabled={isTraining} className="flex items-center space-x-2">
                  <Play className="h-4 w-4" />
                  <span>Start Training</span>
                </Button>

                <Button
                  onClick={pauseTraining}
                  disabled={!isTraining}
                  variant="outline"
                  className="flex items-center space-x-2 bg-transparent"
                >
                  <Pause className="h-4 w-4" />
                  <span>{isPaused ? "Resume" : "Pause"}</span>
                </Button>

                <Button
                  onClick={resetTraining}
                  variant="outline"
                  className="flex items-center space-x-2 bg-transparent"
                >
                  <RotateCcw className="h-4 w-4" />
                  <span>Reset</span>
                </Button>
              </div>

              {isTraining && (
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Episode Progress</span>
                      <span>
                        {episode} / {totalEpisodes}
                      </span>
                    </div>
                    <Progress value={(episode / totalEpisodes) * 100} className="w-full" />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-muted/20 rounded-lg">
                      <p className="text-sm text-muted-foreground">Current Episode</p>
                      <p className="text-2xl font-bold text-primary">{episode}</p>
                    </div>
                    <div className="text-center p-3 bg-muted/20 rounded-lg">
                      <p className="text-sm text-muted-foreground">Cumulative Reward</p>
                      <p className="text-2xl font-bold text-chart-2">{reward.toFixed(1)}</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Live Training Visualization */}
          {isTraining && (
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Training Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={Array.from({ length: Math.floor(episode / 10) + 1 }, (_, i) => ({
                    episode: i * 10,
                    reward: Math.sin(i * 0.1) * 50 + i * 0.5 + Math.random() * 10
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="reward" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Results */}
      {results && (
        <Tabs defaultValue="policy" className="space-y-4">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="policy">Learned Policy</TabsTrigger>
            <TabsTrigger value="training">Training Curve</TabsTrigger>
            <TabsTrigger value="analysis">Policy Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="policy" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="metric-card-1">
                <CardContent className="p-6">
                  <div className="text-center">
                    <Target className="h-8 w-8 text-primary mx-auto mb-2" />
                    <p className="text-sm font-medium text-muted-foreground">Waste Reduction</p>
                    <p className="text-3xl font-bold text-primary">{results.policy.wasteReduction}%</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="metric-card-2">
                <CardContent className="p-6">
                  <div className="text-center">
                    <TrendingUp className="h-8 w-8 text-chart-2 mx-auto mb-2" />
                    <p className="text-sm font-medium text-muted-foreground">Profit Increase</p>
                    <p className="text-3xl font-bold text-chart-2">{results.policy.profitIncrease}%</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="metric-card-3">
                <CardContent className="p-6">
                  <div className="text-center">
                    <Brain className="h-8 w-8 text-chart-3 mx-auto mb-2" />
                    <p className="text-sm font-medium text-muted-foreground">Customer Satisfaction</p>
                    <p className="text-3xl font-bold text-chart-3">
                      {(results.policy.customerSatisfaction * 100).toFixed(0)}%
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="training" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Training Curve Analysis</CardTitle>
                <CardDescription>Reward progression over training episodes</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={results.trainingCurve}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="reward" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analysis" className="space-y-4">
            <Card className="chart-card">
              <CardHeader>
                <CardTitle>Policy Performance Metrics</CardTitle>
                <CardDescription>Optimal pricing actions across different states</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={[
                    { metric: "Waste Reduction", value: results.policy.wasteReduction, color: "#10b981" },
                    { metric: "Profit Increase", value: results.policy.profitIncrease, color: "#3b82f6" },
                    { metric: "Customer Satisfaction", value: results.policy.customerSatisfaction * 100, color: "#f59e0b" },
                    { metric: "Price Stability", value: 85, color: "#8b5cf6" }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip formatter={(value) => [`${value}%`, 'Performance']} />
                    <Bar dataKey="value" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      {results && (
        <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
          <Zap className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800 dark:text-green-200">
            RL training complete! The agent learned a policy achieving {results.policy.wasteReduction}% waste reduction
            and {results.policy.profitIncrease}% profit increase. View the comparison dashboard next.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
