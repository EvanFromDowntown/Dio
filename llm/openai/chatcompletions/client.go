// Package chatcompletions implements llm.Client over the OpenAI Chat
// Completions API (POST {baseURL}/chat/completions), the protocol most
// OpenAI-compatible gateways and open-weight model servers expose.
package chatcompletions

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
	// DefaultBaseURL includes the /v1 prefix, like the openai adapter.
	DefaultBaseURL = "https://api.openai.com/v1"
	// ProviderName tags llm.Reasoning blocks produced by this adapter.
	ProviderName = llm.ProviderChatCompletions
)

type Client struct {
	baseURL         string
	apiKey          string
	httpClient      *http.Client
	header          http.Header
	maxTokensField  string
	streamUsage     bool
	replayReasoning bool
}

var _ llm.Client = (*Client)(nil)

type Option func(*Client)

func WithAPIKey(key string) Option { return func(c *Client) { c.apiKey = key } }

// WithBaseURL points the client at a compatible endpoint; "/chat/completions"
// is appended to it.
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = strings.TrimRight(url, "/") }
}

func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.httpClient = hc } }

func WithHeader(key, value string) Option {
	return func(c *Client) { c.header.Add(key, value) }
}

// WithMaxTokensField sets the request field Request.MaxTokens is sent as.
// The default is "max_completion_tokens"; older servers expect "max_tokens".
func WithMaxTokensField(name string) Option { return func(c *Client) { c.maxTokensField = name } }

// WithStreamUsage controls whether streaming requests send
// stream_options.include_usage (default true).
func WithStreamUsage(enabled bool) Option { return func(c *Client) { c.streamUsage = enabled } }

// WithReasoningReplay controls whether reasoning from earlier assistant turns
// is sent back as reasoning_content (default true). DeepSeek-style thinking
// models expect it during tool-call loops; disable it for servers that reject
// the field.
func WithReasoningReplay(enabled bool) Option {
	return func(c *Client) { c.replayReasoning = enabled }
}

func New(opts ...Option) *Client {
	c := &Client{
		baseURL:         DefaultBaseURL,
		httpClient:      http.DefaultClient,
		header:          http.Header{},
		maxTokensField:  "max_completion_tokens",
		streamUsage:     true,
		replayReasoning: true,
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
		return nil, fmt.Errorf("chatcompletions: decode response: %w", err)
	}
	return convertResponse(&out)
}

func (c *Client) Stream(ctx context.Context, req *llm.Request) (llm.Stream, error) {
	resp, err := c.post(ctx, req, true)
	if err != nil {
		return nil, err
	}
	d := &streamDecoder{}
	return transport.NewStreamWithEOF(resp.Body, d.decode, d.eof), nil
}

func (c *Client) post(ctx context.Context, req *llm.Request, stream bool) (*http.Response, error) {
	body, err := c.encodeRequest(req, stream)
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
	return transport.PostJSON(ctx, c.httpClient, ProviderName, c.baseURL+"/chat/completions", header, body)
}
