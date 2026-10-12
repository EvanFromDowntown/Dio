package llm

import (
	"errors"
	"fmt"
	"net/http"
	"time"
)

// APIError is an error reported by a provider, either as a non-2xx HTTP
// response or as an error event inside a stream (StatusCode 0).
type APIError struct {
	Provider   string
	StatusCode int
	// Type and Code are the provider error type/code, e.g.
	// "rate_limit_error" or "context_length_exceeded".
	Type    string
	Code    string
	Message string
	// RetryAfter is parsed from the Retry-After header when present.
	RetryAfter time.Duration
	// Body is the raw error payload.
	Body []byte
}

func (e *APIError) Error() string {
	msg := e.Message
	if msg == "" {
		msg = http.StatusText(e.StatusCode)
	}
	kind := e.Type
	if e.Code != "" {
		if kind != "" {
			kind += "/"
		}
		kind += e.Code
	}
	switch {
	case e.StatusCode != 0 && kind != "":
		return fmt.Sprintf("%s: HTTP %d (%s): %s", e.Provider, e.StatusCode, kind, msg)
	case e.StatusCode != 0:
		return fmt.Sprintf("%s: HTTP %d: %s", e.Provider, e.StatusCode, msg)
	case kind != "":
		return fmt.Sprintf("%s: %s: %s", e.Provider, kind, msg)
	default:
		return fmt.Sprintf("%s: %s", e.Provider, msg)
	}
}

// Retryable reports whether the request may succeed if retried unchanged:
// rate limits, overload and server errors.
func (e *APIError) Retryable() bool {
	switch e.StatusCode {
	case http.StatusRequestTimeout, http.StatusConflict, http.StatusTooManyRequests:
		return true
	}
	if e.StatusCode >= 500 {
		return true
	}
	switch e.Type {
	case "rate_limit_error", "overloaded_error", "api_error", "server_error":
		return true
	}
	switch e.Code {
	case "rate_limit_exceeded", "server_error":
		return true
	}
	return false
}

// IsRetryable reports whether err wraps a retryable *APIError.
func IsRetryable(err error) bool {
	var apiErr *APIError
	return errors.As(err, &apiErr) && apiErr.Retryable()
}
