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
  Star,
  ArrowRight,
  CheckCircle,
  Users,
  DollarSign,
  TrendingDown,
  Shield,
  Lock,
  Eye,
  Globe,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { UploadSection } from "@/components/upload-section"
import { EDASection } from "@/components/eda-section"
import { MLSection } from "@/components/ml-section"
import { RLSection } from "@/components/rl-section"
import { DashboardSection } from "@/components/dashboard-section"
import { ExportSection } from "@/components/export-section"
import { AnalyticsDashboard } from "@/components/analytics-dashboard"

const tabs = [
  { id: "upload", label: "Upload Data", icon: Upload, description: "Upload your sales data files" },
  { id: "eda", label: "Data Analysis", icon: BarChart3, description: "Explore your data insights" },
  { id: "ml", label: "AI Predictions", icon: Brain, description: "Machine Learning Forecasts" },
  { id: "rl", label: "Price Optimization", icon: Zap, description: "Smart Pricing Engine" },
  { id: "dashboard", label: "Results", icon: TrendingUp, description: "Compare Strategies" },
  { id: "export", label: "Export", icon: Download, description: "Download Reports" },
]

const testimonials = [
  {
    name: "Adebayo Ogundimu",
    role: "Store Manager",
    company: "FreshMart Lagos",
    content:
      "Reduced food waste by 35% and increased profits by 22% in just 3 months. The AI recommendations are spot-on!",
    rating: 5,
    metric: "35% waste reduction",
    image: "/happy-nigerian-store-manager-professional-headshot.jpg",
    location: "Lagos, Nigeria",
  },
  {
    name: "Chioma Okwu",
    role: "Operations Director",
    company: "Global Supermart Nigeria",
    content:
      "The dynamic pricing helped us optimize inventory turnover across 50 stores. We're seeing 18% better margins.",
    rating: 5,
    metric: "18% margin improvement",
    image: "/happy-nigerian-woman-business-professional-headshot.jpg",
    location: "Port Harcourt, Nigeria",
  },
  {
    name: "Grace Okafor",
    role: "Regional Manager",
    company: "MegaMart Chain",
    content: "Easy to use and incredibly powerful. Our team adopted it quickly and results were immediate.",
    rating: 5,
    metric: "Immediate ROI",
    image: "/happy-nigerian-woman-business-professional-headshot.jpg",
    location: "Abuja, Nigeria",
  },
  {
    name: "Emeka Nwosu",
    role: "CEO",
    company: "SuperFresh Markets",
    content: "PriceOptima transformed our pricing strategy. Customer satisfaction up 25% while profits increased.",
    rating: 5,
    metric: "25% satisfaction boost",
    image: "/happy-nigerian-store-manager-professional-headshot.jpg",
    location: "Kano, Nigeria",
  },
  {
    name: "Amina Hassan",
    role: "Store Owner",
    company: "Northern Markets Ltd",
    content: "The waste reduction alone saved us ₦2M annually. Best investment we've made for our business.",
    rating: 5,
    metric: "₦2M saved annually",
    image: "/happy-nigerian-woman-business-professional-headshot.jpg",
    location: "Kaduna, Nigeria",
  },
  {
    name: "Kwame Asante",
    role: "Operations Manager",
    company: "Gold Coast Grocers",
    content: "The real-time analytics help us make instant pricing decisions. Revenue increased by 30%.",
    rating: 5,
    metric: "30% revenue increase",
    image: "/happy-ghanaian-man-business-professional-headshot.jpg",
    location: "Accra, Ghana",
  },
]

const metrics = [
  { value: "35%", label: "Average Waste Reduction", icon: TrendingDown, color: "text-green-600" },
  { value: "22%", label: "Profit Increase", icon: DollarSign, color: "text-blue-600" },
  { value: "500+", label: "Stores Using PriceOptima", icon: Users, color: "text-purple-600" },
  { value: "₦2.5B", label: "Revenue Optimized", icon: TrendingUp, color: "text-orange-600" },
]

const securityFeatures = [
  {
    icon: Shield,
    title: "Enterprise Security",
    description: "Bank-level encryption and ISO 27001 certified infrastructure",
  },
  {
    icon: Lock,
    title: "Data Privacy",
    description: "GDPR & NDPR compliant with zero data sharing to third parties",
  },
  {
    icon: Eye,
    title: "Transparent Analytics",
    description: "Full visibility into how your data is processed and analyzed",
  },
]

export default function HomePage() {
  const [activeTab, setActiveTab] = useState("upload")
  const [uploadedData, setUploadedData] = useState(null)
  const [showApp, setShowApp] = useState(false)
  const [showAnalytics, setShowAnalytics] = useState(false)

  const renderTabContent = () => {
    switch (activeTab) {
      case "upload":
        return <UploadSection onDataUploaded={setUploadedData} />
      case "eda":
        return <EDASection data={uploadedData} />
      case "ml":
        return <MLSection data={uploadedData} />
      case "rl":
        return <RLSection data={uploadedData} />
      case "dashboard":
        return <DashboardSection data={uploadedData} />
      case "export":
        return <ExportSection data={uploadedData} />
      default:
        return <UploadSection onDataUploaded={setUploadedData} />
    }
  }

  const getStepStatus = (stepIndex: number) => {
    const currentIndex = tabs.findIndex((tab) => tab.id === activeTab)
    if (stepIndex < currentIndex) return "completed"
    if (stepIndex === currentIndex) return "active"
    return "inactive"
  }

  if (showAnalytics) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card/50 backdrop-blur-sm">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Button
                  variant="ghost"
                  onClick={() => setShowAnalytics(false)}
                  className="text-muted-foreground hover:text-foreground"
                >
                  ← Back to Home
                </Button>
                <div>
                  <h1 className="text-2xl font-bold text-primary">PriceOptima</h1>
                  <p className="text-sm text-muted-foreground">Real-Time Analytics Dashboard</p>
                </div>
              </div>
            </div>
          </div>
        </header>
        <main className="container mx-auto px-6 py-8">
          <AnalyticsDashboard />
        </main>
      </div>
    )
  }

  if (!showApp) {
    return (
      <div className="min-h-screen bg-background">
        <section className="hero-gradient text-white py-20">
          <div className="container mx-auto px-6 text-center">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
              <h1 className="text-5xl md:text-6xl font-bold mb-6 text-balance">
                The complete platform to optimize retail pricing
              </h1>
              <p className="text-xl md:text-2xl mb-8 text-white/90 max-w-3xl mx-auto text-pretty">
                AI-powered dynamic pricing that reduces waste, increases profits, and optimizes inventory for
                supermarkets and retail stores worldwide.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  size="lg"
                  className="bg-white text-primary hover:bg-white/90 text-lg px-8 py-4 glow-button"
                  onClick={() => setShowApp(true)}
                >
                  Start Free Analysis
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  className="border-white text-white hover:bg-white/10 text-lg px-8 py-4 bg-transparent"
                  onClick={() => setShowAnalytics(true)}
                >
                  View Live Analytics
                </Button>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="py-16 bg-background">
          <div className="container mx-auto px-6">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-foreground mb-4">Trusted by retailers worldwide</h2>
              <p className="text-lg text-muted-foreground">Real results from real businesses</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {metrics.map((metric, index) => {
                const Icon = metric.icon
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                  >
                    <Card className="metric-card text-center p-6 hover:scale-105 transition-transform">
                      <Icon className={`h-8 w-8 mx-auto mb-3 ${metric.color}`} />
                      <div className="text-3xl font-bold text-foreground mb-1">{metric.value}</div>
                      <div className="text-sm text-muted-foreground">{metric.label}</div>
                    </Card>
                  </motion.div>
                )
              })}
            </div>
          </div>
        </section>

        <section className="py-20 bg-muted/30">
          <div className="container mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-foreground mb-4">Why retailers choose PriceOptima</h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Transform your pricing strategy with AI-powered insights that deliver real business results
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              <Card className="p-8 text-center hover:shadow-lg transition-shadow">
                <div className="mb-6">
                  <img
                    src="/modern-supermarket-fresh-produce-section-with-digi.jpg"
                    alt="Smart pricing in action"
                    className="w-full h-48 object-cover rounded-lg"
                  />
                </div>
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <TrendingDown className="h-8 w-8 text-primary" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Reduce Waste by 35%</h3>
                <p className="text-muted-foreground mb-6">
                  Smart pricing algorithms identify optimal price points to move inventory before expiration,
                  dramatically reducing food waste and environmental impact.
                </p>
                <Badge variant="secondary" className="text-green-600 bg-green-50">
                  Proven Results
                </Badge>
              </Card>

              <Card className="p-8 text-center hover:shadow-lg transition-shadow">
                <div className="mb-6">
                  <img
                    src="/happy-store-manager-looking-at-profit-dashboard-on.jpg"
                    alt="Profit optimization dashboard"
                    className="w-full h-48 object-cover rounded-lg"
                  />
                </div>
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <DollarSign className="h-8 w-8 text-primary" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Increase Profits by 22%</h3>
                <p className="text-muted-foreground mb-6">
                  Dynamic pricing optimization ensures you're maximizing revenue while maintaining competitive
                  positioning in the market across all product categories.
                </p>
                <Badge variant="secondary" className="text-blue-600 bg-blue-50">
                  Revenue Growth
                </Badge>
              </Card>

              <Card className="p-8 text-center hover:shadow-lg transition-shadow">
                <div className="mb-6">
                  <img
                    src="/ai-analytics-dashboard-with-charts-and-graphs-on-c.jpg"
                    alt="AI-powered analytics"
                    className="w-full h-48 object-cover rounded-lg"
                  />
                </div>
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Brain className="h-8 w-8 text-primary" />
                </div>
                <h3 className="text-2xl font-bold mb-4">AI-Powered Insights</h3>
                <p className="text-muted-foreground mb-6">
                  Advanced machine learning analyzes sales patterns, seasonality, and market trends to recommend optimal
                  pricing strategies with 95% accuracy.
                </p>
                <Badge variant="secondary" className="text-purple-600 bg-purple-50">
                  Smart Technology
                </Badge>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-20 bg-background">
          <div className="container mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-foreground mb-4">Your data is safe with us</h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                We take privacy seriously and protect your business information
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {securityFeatures.map((feature, index) => {
                const Icon = feature.icon
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                  >
                    <Card className="p-8 text-center border-primary/20 bg-primary/5 hover:shadow-lg transition-shadow">
                      <div className="w-16 h-16 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-6">
                        <Icon className="h-8 w-8 text-primary" />
                      </div>
                      <h3 className="text-2xl font-bold mb-4 text-white">{feature.title}</h3>
                      <p className="text-white/90 text-lg font-medium">{feature.description}</p>
                    </Card>
                  </motion.div>
                )
              })}
            </div>

            <div className="mt-12 text-center">
              <div className="flex items-center justify-center space-x-8 mb-6">
                <Badge variant="outline" className="text-white border-white px-4 py-2 text-lg font-bold">
                  <Shield className="h-4 w-4 mr-2" />
                  Secure & Private
                </Badge>
                <Badge variant="outline" className="text-white border-white px-4 py-2 text-lg font-bold">
                  <Globe className="h-4 w-4 mr-2" />
                  GDPR Compliant
                </Badge>
              </div>

              <Card className="max-w-2xl mx-auto p-6 bg-card/80 backdrop-blur-sm">
                <div className="flex items-center space-x-4 mb-4">
                  <img
                    src="/happy-nigerian-man-store-manager-headshot.jpg"
                    alt="Tunde Adebayo"
                    className="w-16 h-16 rounded-full object-cover"
                  />
                  <div className="text-left">
                    <div className="font-semibold text-foreground text-lg">Tunde Adebayo</div>
                    <div className="text-muted-foreground">IT Manager, ShopRite Nigeria</div>
                    <div className="flex text-yellow-400">
                      {[...Array(5)].map((_, i) => (
                        <Star key={i} className="h-4 w-4 fill-current" />
                      ))}
                    </div>
                  </div>
                </div>
                <p className="text-foreground text-lg italic">
                  "We were initially concerned about data security, but PriceOptima's transparent approach and clear
                  privacy policies gave us confidence. Our customer data stays private, and we have full control over
                  what information we share. It's exactly what we needed for our business."
                </p>
              </Card>

              <p className="text-lg text-white/90 max-w-2xl mx-auto font-medium mt-6">
                Your sales data never leaves your control. We process information securely and never share customer
                details with anyone.
              </p>
            </div>
          </div>
        </section>

        <section className="py-20 bg-muted/30">
          <div className="container mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-foreground mb-4">Loved by retailers worldwide</h2>
              <p className="text-xl text-muted-foreground">
                See how PriceOptima is transforming businesses across the globe
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {testimonials.slice(0, 6).map((testimonial, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                >
                  <Card className="testimonial-card p-6 h-full hover:shadow-lg transition-shadow">
                    <div className="flex items-center mb-4">
                      {[...Array(testimonial.rating)].map((_, i) => (
                        <Star key={i} className="h-5 w-5 text-yellow-400 fill-current" />
                      ))}
                    </div>
                    <p className="text-foreground mb-6 italic">"{testimonial.content}"</p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <img
                          src={testimonial.image || "/placeholder.svg"}
                          alt={testimonial.name}
                          className="w-12 h-12 rounded-full object-cover"
                        />
                        <div>
                          <div className="font-semibold text-foreground">{testimonial.name}</div>
                          <div className="text-sm text-muted-foreground">{testimonial.role}</div>
                          <div className="text-sm text-muted-foreground">{testimonial.company}</div>
                          <div className="text-xs text-muted-foreground flex items-center">
                            <Globe className="h-3 w-3 mr-1" />
                            {testimonial.location}
                          </div>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-primary border-primary">
                        {testimonial.metric}
                      </Badge>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-20 bg-primary text-primary-foreground">
          <div className="container mx-auto px-6 text-center">
            <h2 className="text-4xl font-bold mb-4">Ready to optimize your pricing?</h2>
            <p className="text-xl mb-8 text-primary-foreground/90 max-w-2xl mx-auto">
              Join hundreds of retailers already using PriceOptima to reduce waste and increase profits
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                size="lg"
                className="bg-white text-primary hover:bg-white/90 text-lg px-8 py-4 glow-button"
                onClick={() => setShowApp(true)}
              >
                Start Your Free Analysis
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-white text-white hover:bg-white/10 text-lg px-8 py-4 bg-transparent"
                onClick={() => setShowAnalytics(true)}
              >
                View Live Demo
              </Button>
            </div>
            <p className="text-sm text-primary-foreground/70 mt-4">
              No credit card required • 14-day free trial • Setup in under 5 minutes
            </p>
          </div>
        </section>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                onClick={() => setShowApp(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                ← Back to Home
              </Button>
              <div>
                <h1 className="text-2xl font-bold text-primary">PriceOptima</h1>
                <p className="text-sm text-muted-foreground">AI-Powered Retail Pricing Platform</p>
              </div>
            </div>
            <Button variant="outline" onClick={() => setShowAnalytics(true)} className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>Live Analytics</span>
            </Button>
          </div>
        </div>
      </header>

      <div className="border-b border-border bg-card/30 backdrop-blur-sm py-4">
        <div className="container mx-auto px-6">
          <div className="flex items-center justify-between max-w-4xl mx-auto">
            {tabs.map((tab, index) => {
              const Icon = tab.icon
              const status = getStepStatus(index)

              return (
                <div key={tab.id} className="flex items-center">
                  <div className="flex flex-col items-center">
                    <div className={`progress-step ${status}`}>
                      {status === "completed" ? <CheckCircle className="h-5 w-5" /> : <Icon className="h-5 w-5" />}
                    </div>
                    <div className="text-xs mt-2 text-center max-w-20">
                      <div className="font-medium">{tab.label}</div>
                    </div>
                  </div>
                  {index < tabs.length - 1 && (
                    <div
                      className={`progress-line ${getStepStatus(index + 1) === "completed" ? "completed" : status === "active" ? "active" : ""} w-16 mx-4`}
                    />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="border-b border-border bg-card/30 backdrop-blur-sm">
        <div className="container mx-auto px-6">
          <div className="flex space-x-1 overflow-x-auto py-2">
            {tabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id

              return (
                <Button
                  key={tab.id}
                  variant={isActive ? "default" : "ghost"}
                  className={`relative flex items-center space-x-2 px-4 py-3 rounded-lg transition-all duration-200 whitespace-nowrap ${
                    isActive
                      ? "bg-primary text-primary-foreground shadow-lg"
                      : "hover:bg-accent hover:text-accent-foreground"
                  }`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <Icon className="h-4 w-4" />
                  <div className="text-left">
                    <div className="font-medium">{tab.label}</div>
                    <div className="text-xs opacity-80">{tab.description}</div>
                  </div>
                </Button>
              )
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="min-h-[600px]"
        >
          {renderTabContent()}
        </motion.div>
      </main>
    </div>
  )
}
