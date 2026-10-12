// Package llm defines a provider-agnostic interface for calling large language
// models. Concrete providers live in sub-packages (llm/openai, llm/anthropic)
// and translate these types to and from their wire protocols.
package llm

import "strings"

// Role identifies the author of a message. System instructions are carried
// by Request.System rather than by a message role, and tool results are sent
// as ContentToolResult blocks inside a RoleUser message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// ContentType discriminates the payload of a ContentBlock.
type ContentType string

const (
	ContentText       ContentType = "text"
	ContentImage      ContentType = "image"
	ContentToolCall   ContentType = "tool_call"
	ContentToolResult ContentType = "tool_result"
	ContentReasoning  ContentType = "reasoning"
)

// ContentBlock is one piece of message content. Exactly one payload field,
// selected by Type, is meaningful.
type ContentBlock struct {
	Type       ContentType `json:"type"`
	Text       string      `json:"text,omitempty"`
	Image      *Image      `json:"image,omitempty"`
	ToolCall   *ToolCall   `json:"tool_call,omitempty"`
	ToolResult *ToolResult `json:"tool_result,omitempty"`
	Reasoning  *Reasoning  `json:"reasoning,omitempty"`
}

// Image is an image input, given either by URL or as base64 data.
type Image struct {
	URL       string `json:"url,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

// Reasoning is model "thinking" output. It is opaque provider state that must
// be replayed verbatim to the provider that produced it, so adapters drop
// reasoning blocks whose Provider does not match their own.
type Reasoning struct {
	Provider string `json:"provider"`
	// ID is the provider item id (OpenAI reasoning items).
	ID string `json:"id,omitempty"`
	// Text is the human-readable thinking or reasoning summary.
	Text string `json:"text,omitempty"`
	// Signature authenticates Text (Anthropic thinking blocks).
	Signature string `json:"signature,omitempty"`
	// Encrypted holds opaque reasoning content (OpenAI encrypted_content,
	// Anthropic redacted_thinking data).
	Encrypted string `json:"encrypted,omitempty"`
}

// Message is a single conversation turn.
type Message struct {
	Role    Role           `json:"role"`
	Content []ContentBlock `json:"content"`
}

// Text returns the concatenation of all text blocks in the message.
func (m Message) Text() string {
	var b strings.Builder
	for _, c := range m.Content {
		if c.Type == ContentText {
			b.WriteString(c.Text)
		}
	}
	return b.String()
}

// ToolCalls returns the tool calls requested in the message, in order.
func (m Message) ToolCalls() []ToolCall {
	var calls []ToolCall
	for _, c := range m.Content {
		if c.Type == ContentToolCall && c.ToolCall != nil {
			calls = append(calls, *c.ToolCall)
		}
	}
	return calls
}

func TextBlock(text string) ContentBlock {
	return ContentBlock{Type: ContentText, Text: text}
}

func ImageURLBlock(url string) ContentBlock {
	return ContentBlock{Type: ContentImage, Image: &Image{URL: url}}
}

func ImageDataBlock(mediaType, base64Data string) ContentBlock {
	return ContentBlock{Type: ContentImage, Image: &Image{MediaType: mediaType, Data: base64Data}}
}

func ToolCallBlock(call ToolCall) ContentBlock {
	return ContentBlock{Type: ContentToolCall, ToolCall: &call}
}

func ToolResultBlock(result ToolResult) ContentBlock {
	return ContentBlock{Type: ContentToolResult, ToolResult: &result}
}

// UserText builds a user message containing a single text block.
func UserText(text string) Message {
	return Message{Role: RoleUser, Content: []ContentBlock{TextBlock(text)}}
}

// AssistantText builds an assistant message containing a single text block.
func AssistantText(text string) Message {
	return Message{Role: RoleAssistant, Content: []ContentBlock{TextBlock(text)}}
}

// ToolResults builds the user message that answers a batch of tool calls.
func ToolResults(results ...ToolResult) Message {
	blocks := make([]ContentBlock, len(results))
	for i, r := range results {
		blocks[i] = ToolResultBlock(r)
	}
	return Message{Role: RoleUser, Content: blocks}
}
