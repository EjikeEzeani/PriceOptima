/**
 * Image Management Utility for PriceOptima
 * Handles multiple image sources, fallbacks, and local image management
 */

export interface ImageConfig {
  name: string;
  description: string;
  current: string;
  alternatives: string[];
  local_path: string;
  ethnicity?: string;
  gender?: string;
  region?: string;
}

export interface ImageServices {
  primary: string;
  fallback: string;
  local: boolean;
}

export interface ImageConfigData {
  testimonials: Record<string, ImageConfig>;
  features: Record<string, ImageConfig>;
  image_services: ImageServices;
}

class ImageManager {
  private config: ImageConfigData | null = null;
  private fallbackImages: Record<string, string> = {};

  constructor() {
    this.loadConfig();
  }

  private async loadConfig() {
    try {
      const response = await fetch('/images/image-config.json');
      this.config = await response.json();
    } catch (error) {
      console.warn('Could not load image config, using defaults');
      this.config = null;
    }
  }

  /**
   * Get image URL with fallback support
   */
  getImageUrl(key: string, type: 'testimonials' | 'features' = 'testimonials'): string {
    if (!this.config) {
      return this.getDefaultImage(key);
    }

    const imageConfig = this.config[type][key];
    if (!imageConfig) {
      return this.getDefaultImage(key);
    }

    // Try local image first if available
    if (this.config.image_services.local) {
      return imageConfig.local_path;
    }

    // Use current image
    return imageConfig.current;
  }

  /**
   * Get alternative image URLs
   */
  getAlternativeImages(key: string, type: 'testimonials' | 'features' = 'testimonials'): string[] {
    if (!this.config) {
      return [this.getDefaultImage(key)];
    }

    const imageConfig = this.config[type][key];
    if (!imageConfig) {
      return [this.getDefaultImage(key)];
    }

    return imageConfig.alternatives;
  }

  /**
   * Get image with error handling and fallback
   */
  getImageWithFallback(key: string, type: 'testimonials' | 'features' = 'testimonials'): {
    primary: string;
    fallbacks: string[];
  } {
    const primary = this.getImageUrl(key, type);
    const fallbacks = this.getAlternativeImages(key, type);
    
    return {
      primary,
      fallbacks: fallbacks.filter(url => url !== primary)
    };
  }

  /**
   * Get all testimonial images
   */
  getAllTestimonialImages(): Record<string, ImageConfig> {
    if (!this.config) {
      return {};
    }
    return this.config.testimonials;
  }

  /**
   * Get all feature images
   */
  getAllFeatureImages(): Record<string, ImageConfig> {
    if (!this.config) {
      return {};
    }
    return this.config.features;
  }

  /**
   * Update image configuration
   */
  updateImageConfig(key: string, type: 'testimonials' | 'features', updates: Partial<ImageConfig>) {
    if (!this.config) {
      return;
    }

    if (this.config[type][key]) {
      this.config[type][key] = { ...this.config[type][key], ...updates };
    }
  }

  /**
   * Switch to local images
   */
  switchToLocalImages() {
    if (!this.config) {
      return;
    }
    this.config.image_services.local = true;
  }

  /**
   * Switch to remote images
   */
  switchToRemoteImages() {
    if (!this.config) {
      return;
    }
    this.config.image_services.local = false;
  }

  /**
   * Get default image for fallback
   */
  private getDefaultImage(key: string): string {
    const defaults: Record<string, string> = {
      'marcus-johnson': 'https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
      'keisha-williams': 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
      'jasmine-davis': 'https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
      'david-thompson': 'https://images.pexels.com/photos/1040880/pexels-photo-1040880.jpeg?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
      'michelle-brown': 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
      'james-wilson': 'https://images.pexels.com/photos/1040881/pexels-photo-1040881.jpeg?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
    };
    
    return defaults[key] || 'https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?w=150&h=150&fit=crop&crop=face&auto=format&q=80';
  }
}

// Export singleton instance
export const imageManager = new ImageManager();

// Export utility functions
export const getTestimonialImage = (key: string) => imageManager.getImageUrl(key, 'testimonials');
export const getFeatureImage = (key: string) => imageManager.getImageUrl(key, 'features');
export const getImageWithFallback = (key: string, type: 'testimonials' | 'features' = 'testimonials') => 
  imageManager.getImageWithFallback(key, type);
