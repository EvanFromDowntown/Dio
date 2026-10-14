package webui

import (
	"time"

	"github.com/EvanFromDowntown/Dio/internal/cli"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/session"
)

// Item is one row of a rendered conversation, shared by the history view
// and the live stream so the page draws both the same way.
type Item struct {
	Kind string `json:"kind"` // user | assistant | reasoning | tool_call | tool_result
	Text string `json:"text,omitempty"`
	// Tool rows.
	ToolID  string `json:"tool_id,omitempty"`
	Name    string `json:"name,omitempty"`
	Args    string `json:"args,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
	Bytes   int    `json:"bytes,omitempty"`
	Preview string `json:"preview,omitempty"`
}

type SessionSummary struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	Provider     string    `json:"provider"`
	Model        string    `json:"model"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
	MessageCount int       `json:"message_count"`
	Usage        llm.Usage `json:"usage"`
}

func summarize(info session.Info) SessionSummary {
	return SessionSummary{
		ID: info.ID, Title: info.Title, Provider: info.Provider, Model: info.Model,
		CreatedAt: info.CreatedAt, UpdatedAt: info.UpdatedAt,
		MessageCount: info.MessageCount, Usage: info.Usage,
	}
}

func toolCallItem(c llm.ToolCall) Item {
	return Item{Kind: "tool_call", ToolID: c.ID, Name: c.Name, Args: cli.SummarizeArgs(c.Arguments, 120)}
}

func toolResultItem(name string, r llm.ToolResult) Item {
	it := Item{Kind: "tool_result", ToolID: r.ToolCallID, Name: name, IsError: r.IsError, Bytes: len(r.Content)}
	if r.IsError {
		it.Preview = cli.Preview(r.Content, 160)
	} else {
		it.Preview = cli.Preview(r.Content, 80)
	}
	return it
}

// historyItems renders stored messages for display.
func historyItems(msgs []llm.Message) []Item {
	items := []Item{}
	toolNames := map[string]string{}
	for _, m := range msgs {
		for _, b := range m.Content {
			switch {
			case b.Type == llm.ContentText && b.Text != "":
				kind := "user"
				if m.Role == llm.RoleAssistant {
					kind = "assistant"
				}
				items = append(items, Item{Kind: kind, Text: b.Text})
			case b.Type == llm.ContentReasoning && b.Reasoning != nil && b.Reasoning.Text != "":
				items = append(items, Item{Kind: "reasoning", Text: b.Reasoning.Text})
			case b.Type == llm.ContentToolCall && b.ToolCall != nil:
				toolNames[b.ToolCall.ID] = b.ToolCall.Name
				items = append(items, toolCallItem(*b.ToolCall))
			case b.Type == llm.ContentToolResult && b.ToolResult != nil:
				items = append(items, toolResultItem(toolNames[b.ToolResult.ToolCallID], *b.ToolResult))
			case b.Type == llm.ContentImage:
				items = append(items, Item{Kind: "user", Text: "[图片]"})
			}
		}
	}
	return items
}
