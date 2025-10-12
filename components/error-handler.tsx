'use client';

import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { AlertCircle, RefreshCw, Wifi, WifiOff } from 'lucide-react';

interface ErrorState {
  message: string;
  type: 'network' | 'server' | 'validation' | 'unknown';
  retryable: boolean;
  timestamp: number;
}

interface ErrorContextType {
  error: ErrorState | null;
  setError: (error: ErrorState | null) => void;
  clearError: () => void;
  isRetrying: boolean;
  retry: () => void;
}

const ErrorContext = createContext<ErrorContextType | undefined>(undefined);

export const useError = () => {
  const context = useContext(ErrorContext);
  if (!context) {
    throw new Error('useError must be used within an ErrorProvider');
  }
  return context;
};

interface ErrorProviderProps {
  children: ReactNode;
}

export const ErrorProvider: React.FC<ErrorProviderProps> = ({ children }) => {
  const [error, setError] = useState<ErrorState | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);

  const clearError = () => setError(null);

  const retry = async () => {
    if (!error?.retryable) return;
    
    setIsRetrying(true);
    clearError();
    
    // Wait a moment before allowing new requests
    setTimeout(() => {
      setIsRetrying(false);
    }, 2000);
  };

  return (
    <ErrorContext.Provider value={{ error, setError, clearError, isRetrying, retry }}>
      {children}
      {error && <ErrorDisplay />}
    </ErrorContext.Provider>
  );
};

const ErrorDisplay: React.FC = () => {
  const { error, clearError, retry, isRetrying } = useError();

  if (!error) return null;

  const getErrorIcon = () => {
    switch (error.type) {
      case 'network':
        return <WifiOff className="h-4 w-4" />;
      case 'server':
        return <AlertCircle className="h-4 w-4" />;
      default:
        return <AlertCircle className="h-4 w-4" />;
    }
  };

  const getErrorTitle = () => {
    switch (error.type) {
      case 'network':
        return 'Connection Error';
      case 'server':
        return 'Server Error';
      case 'validation':
        return 'Validation Error';
      default:
        return 'Error';
    }
  };

  const getErrorDescription = () => {
    switch (error.type) {
      case 'network':
        return 'Unable to connect to the server. This might be due to network issues or the server being temporarily unavailable.';
      case 'server':
        return 'The server encountered an error while processing your request. Please try again.';
      case 'validation':
        return error.message;
      default:
        return error.message;
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50 max-w-md">
      <Alert variant="destructive">
        <div className="flex items-start space-x-2">
          {getErrorIcon()}
          <div className="flex-1">
            <h4 className="font-semibold">{getErrorTitle()}</h4>
            <AlertDescription className="mt-1">
              {getErrorDescription()}
            </AlertDescription>
            <div className="flex space-x-2 mt-3">
              {error.retryable && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={retry}
                  disabled={isRetrying}
                  className="text-xs"
                >
                  {isRetrying ? (
                    <>
                      <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                      Retrying...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Retry
                    </>
                  )}
                </Button>
              )}
              <Button
                size="sm"
                variant="ghost"
                onClick={clearError}
                className="text-xs"
              >
                Dismiss
              </Button>
            </div>
          </div>
        </div>
      </Alert>
    </div>
  );
};

// Utility function to create error states
export const createError = (
  message: string,
  type: ErrorState['type'] = 'unknown',
  retryable: boolean = true
): ErrorState => ({
  message,
  type,
  retryable,
  timestamp: Date.now(),
});

// Utility function to handle API errors
export const handleApiError = (error: Error, setError: (error: ErrorState | null) => void) => {
  console.error('API Error:', error);
  
  let errorType: ErrorState['type'] = 'unknown';
  let retryable = true;
  
  if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
    errorType = 'network';
  } else if (error.message.includes('HTTP 5')) {
    errorType = 'server';
  } else if (error.message.includes('HTTP 4')) {
    errorType = 'validation';
    retryable = false;
  }
  
  setError(createError(error.message, errorType, retryable));
};
