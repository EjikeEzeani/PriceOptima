"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { CheckCircle, XCircle, AlertTriangle, RefreshCw } from "lucide-react"

interface DebugPanelProps {
  onTestComplete?: (results: any) => void
}

export function DebugPanel({ onTestComplete }: DebugPanelProps) {
  const [tests, setTests] = useState<Record<string, any>>({})
  const [isRunning, setIsRunning] = useState(false)
  const [enabled, setEnabled] = useState(false)
  const API_URL = process.env.NEXT_PUBLIC_API_URL || '/api'

  const runTest = async (testName: string, testFn: () => Promise<any>, timeout = 10000) => {
    try {
      setTests(prev => ({ ...prev, [testName]: { status: 'running', message: 'Testing...' } }))
      
      // Add timeout to prevent hanging
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Test timeout')), timeout)
      )
      
      const result = await Promise.race([testFn(), timeoutPromise])
      setTests(prev => ({ ...prev, [testName]: { status: 'success', message: 'Passed', data: result } }))
      return result
    } catch (error: any) {
      setTests(prev => ({ ...prev, [testName]: { status: 'error', message: error.message, error } }))
      throw error
    }
  }

  const runAllTests = async () => {
    setIsRunning(true)
    setTests({})

    try {
      // Run basic tests in parallel for speed
      const basicTests = [
        // Test 1: Backend Health Check
        runTest('Backend Health', async () => {
          const response = await fetch(`${API_URL}/health`)
          if (!response.ok) throw new Error(`HTTP ${response.status}`)
          return await response.json()
        }),

        // Test 2: CORS Check
        runTest('CORS Check', async () => {
          const response = await fetch(`${API_URL}/health`, {
            method: 'OPTIONS'
          })
          return { status: response.status, headers: Object.fromEntries(response.headers.entries()) }
        }),

        // Test 3: API Client Test
        runTest('API Client Test', async () => {
          const { apiClient } = await import('@/lib/api')
          return await apiClient.healthCheck()
        })
      ]

      // Run basic tests in parallel
      await Promise.allSettled(basicTests)

      // Only run heavy tests if basic tests pass
      const hasErrors = Object.values(tests).some(test => test.status === 'error')
      
      if (!hasErrors) {
        // Test 4: Quick File Upload Test (lightweight)
        await runTest('File Upload Test', async () => {
          const testData = new Blob(['date,product,category,price,quantity,revenue\n2024-01-01,Test Product,Test Category,100,10,1000'], { type: 'text/csv' })
          const file = new File([testData], 'test.csv', { type: 'text/csv' })
          
          const { uploadData } = await import('@/lib/api')
          return await uploadData(file)
        })
      }

      onTestComplete?.(tests)
    } catch (error) {
      console.error('Debug tests failed:', error)
    } finally {
      setIsRunning(false)
    }
  }

  const runQuickTest = async () => {
    setIsRunning(true)
    setTests({})

    try {
      // Only test backend health - fastest possible test
      await runTest('Backend Health', async () => {
        const response = await fetch(`${API_URL}/health`)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return await response.json()
      })

      onTestComplete?.(tests)
    } catch (error) {
      console.error('Quick test failed:', error)
    } finally {
      setIsRunning(false)
    }
  }

  const runHeavyTests = async () => {
    setIsRunning(true)
    
    try {
      // Test 5: EDA Analysis Test (only if data is uploaded) - 30s timeout
      await runTest('EDA Analysis Test', async () => {
        const { runEDA } = await import('@/lib/api')
        return await runEDA()
      }, 30000)

      // Test 6: ML Model Test (only if data is uploaded) - 60s timeout
      await runTest('ML Model Test', async () => {
        const { trainML } = await import('@/lib/api')
        return await trainML('random_forest')
      }, 60000)

      onTestComplete?.(tests)
    } catch (error) {
      console.error('Heavy tests failed:', error)
    } finally {
      setIsRunning(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error': return <XCircle className="h-4 w-4 text-red-500" />
      case 'running': return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />
      default: return <AlertTriangle className="h-4 w-4 text-yellow-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'success': return <Badge variant="default" className="bg-green-500">Passed</Badge>
      case 'error': return <Badge variant="destructive">Failed</Badge>
      case 'running': return <Badge variant="secondary">Running</Badge>
      default: return <Badge variant="outline">Pending</Badge>
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <RefreshCw className="h-5 w-5" />
          <span>Debug Panel</span>
        </CardTitle>
        <CardDescription>
          Test application components. Use "Basic Tests" for quick health checks, "Heavy Tests" for full analysis.
          <br />
          <span className="text-xs">Toggle debug on to enable tests. Leave off during normal EDA/ML use.</span>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <Button variant={enabled ? 'default' : 'outline'} size="sm" onClick={() => setEnabled(!enabled)}>
            {enabled ? 'Debug: ON' : 'Debug: OFF'}
          </Button>
        </div>

        {!enabled && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Debug is OFF. Turn it ON only when you need to run tests.
            </AlertDescription>
          </Alert>
        )}

        {enabled && (
        <div className="flex flex-wrap gap-2">
          <Button onClick={runQuickTest} disabled={isRunning} variant="secondary" className="flex items-center space-x-2">
            <RefreshCw className={`h-4 w-4 ${isRunning ? 'animate-spin' : ''}`} />
            <span>Quick Test</span>
          </Button>
          <Button onClick={runAllTests} disabled={isRunning} className="flex items-center space-x-2">
            <RefreshCw className={`h-4 w-4 ${isRunning ? 'animate-spin' : ''}`} />
            <span>{isRunning ? 'Running Tests...' : 'Basic Tests'}</span>
          </Button>
          <Button onClick={runHeavyTests} disabled={isRunning} variant="outline" className="flex items-center space-x-2">
            <RefreshCw className={`h-4 w-4 ${isRunning ? 'animate-spin' : ''}`} />
            <span>Heavy Tests</span>
          </Button>
        </div>
        )}

        {enabled && (
        <div className="space-y-3">
          {Object.entries(tests).map(([testName, test]) => (
            <div key={testName} className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center space-x-3">
                {getStatusIcon(test.status)}
                <div>
                  <div className="font-medium">{testName}</div>
                  <div className="text-sm text-muted-foreground">{test.message}</div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {getStatusBadge(test.status)}
                {test.data && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => console.log(`${testName} data:`, test.data)}
                  >
                    View Data
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
        )}

        {enabled && Object.keys(tests).length === 0 && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Choose a test mode: <strong>Quick Test</strong> (fastest), <strong>Basic Tests</strong> (comprehensive), or <strong>Heavy Tests</strong> (full analysis)
            </AlertDescription>
          </Alert>
        )}

        {enabled && isRunning && (
          <Alert>
            <RefreshCw className="h-4 w-4 animate-spin" />
            <AlertDescription>
              Running tests... This may take a few moments for heavy tests.
            </AlertDescription>
          </Alert>
        )}

        {enabled && Object.keys(tests).length > 0 && (
          <div className="mt-4 p-3 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">Test Summary:</h4>
            <div className="text-sm space-y-1">
              <div>Total Tests: {Object.keys(tests).length}</div>
              <div>Passed: {Object.values(tests).filter(t => t.status === 'success').length}</div>
              <div>Failed: {Object.values(tests).filter(t => t.status === 'error').length}</div>
              <div>Running: {Object.values(tests).filter(t => t.status === 'running').length}</div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
