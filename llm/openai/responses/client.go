// Package responses implements llm.Client over the OpenAI Responses API
// (POST {baseURL}/responses).
package responses

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
	// DefaultBaseURL includes the /v1 prefix, matching the official SDKs'
	// OPENAI_BASE_URL convention.
	DefaultBaseURL = "https://api.openai.com/v1"
	// ProviderName tags llm.Reasoning blocks produced by this adapter.
	ProviderName = llm.ProviderResponses
)

type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	header     http.Header
}

var _ llm.Client = (*Client)(nil)

type Option func(*Client)

func WithAPIKey(key string) Option { return func(c *Client) { c.apiKey = key } }

// WithBaseURL points the client at a compatible endpoint; "/responses" is
// appended to it.
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = strings.TrimRight(url, "/") }
}

func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.httpClient = hc } }

// WithHeader adds a header to every request (e.g. OpenAI-Organization or a
// gateway routing header).
func WithHeader(key, value string) Option {
	return func(c *Client) { c.header.Add(key, value) }
}

func New(opts ...Option) *Client {
	c := &Client{baseURL: DefaultBaseURL, httpClient: http.DefaultClient, header: http.Header{}}
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
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}
	return convertResponse(&out)
}

func (c *Client) Stream(ctx context.Context, req *llm.Request) (llm.Stream, error) {
	resp, err := c.post(ctx, req, true)
	if err != nil {
		return nil, err
	}
	return transport.NewStream(resp.Body, newStreamDecoder().decode), nil
}

func (c *Client) post(ctx context.Context, req *llm.Request, stream bool) (*http.Response, error) {
	body, err := encodeRequest(req, stream)
	if err != nil {
		return nil, err
	}
	header := c.header.Clone()
	if c.apiKey != "" {
		header.Set("Authorization", "Bearer "+c.apiKey)
	}
	if stream {
		header.Set("Accept", "text/event-stream")
	}
	return transport.PostJSON(ctx, c.httpClient, ProviderName, c.baseURL+"/responses", header, body)
}
