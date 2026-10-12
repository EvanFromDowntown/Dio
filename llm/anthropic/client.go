// Package anthropic implements llm.Client over the Anthropic Messages API
// (POST {baseURL}/v1/messages).
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/EvanFromDowntown/Dio/internal/transport"
	"github.com/EvanFromDowntown/Dio/llm"
)

const (
	// DefaultBaseURL excludes the /v1 prefix, matching the official SDKs'
	// ANTHROPIC_BASE_URL convention.
	DefaultBaseURL   = "https://api.anthropic.com"
	APIVersion       = "2023-06-01"
	DefaultMaxTokens = 4096
	// ProviderName tags llm.Reasoning blocks produced by this adapter.
	ProviderName = "anthropic"
)

type Client struct {
	baseURL          string
	apiKey           string
	authToken        string
	httpClient       *http.Client
	header           http.Header
	defaultMaxTokens int
}

var _ llm.Client = (*Client)(nil)

type Option func(*Client)

// WithAPIKey authenticates with the x-api-key header.
func WithAPIKey(key string) Option { return func(c *Client) { c.apiKey = key } }

// WithAuthToken authenticates with "Authorization: Bearer", as some gateways
// require.
func WithAuthToken(token string) Option { return func(c *Client) { c.authToken = token } }

// WithBaseURL points the client at a compatible endpoint; "/v1/messages" is
// appended to it.
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = strings.TrimRight(url, "/") }
}

func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.httpClient = hc } }

// WithHeader adds a header to every request (e.g. anthropic-beta).
func WithHeader(key, value string) Option {
	return func(c *Client) { c.header.Add(key, value) }
}

// WithDefaultMaxTokens sets max_tokens for requests that leave MaxTokens
// zero; the Messages API requires the field.
func WithDefaultMaxTokens(n int) Option { return func(c *Client) { c.defaultMaxTokens = n } }

func New(opts ...Option) *Client {
	c := &Client{
		baseURL:          DefaultBaseURL,
		httpClient:       http.DefaultClient,
		header:           http.Header{},
		defaultMaxTokens: DefaultMaxTokens,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

func (c *Client) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	resp, err := c.post(ctx, req, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}
	return convertResponse(&out), nil
}

func (c *Client) Stream(ctx context.Context, req *llm.Request) (llm.Stream, error) {
	resp, err := c.post(ctx, req, true)
	if err != nil {
		return nil, err
	}
	return transport.NewStream(resp.Body, newStreamDecoder().decode), nil
}

func (c *Client) post(ctx context.Context, req *llm.Request, stream bool) (*http.Response, error) {
	body, err := encodeRequest(req, c.defaultMaxTokens, stream)
	if err != nil {
		return nil, err
	}
	header := c.header.Clone()
	header.Set("anthropic-version", APIVersion)
	if c.apiKey != "" {
		header.Set("x-api-key", c.apiKey)
	}
	if c.authToken != "" {
		header.Set("Authorization", "Bearer "+c.authToken)
	}
	if stream {
		header.Set("Accept", "text/event-stream")
	}
	return transport.PostJSON(ctx, c.httpClient, ProviderName, c.baseURL+"/v1/messages", header, body)
}
