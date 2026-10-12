package llm

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestMessageJSONRoundTrip(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []ContentBlock{
			TextBlock("hi"),
			TextBlock(""),
			ImageURLBlock("https://x/a.png"),
			ImageDataBlock("image/png", "AAAA"),
			ToolResultBlock(ToolResult{ToolCallID: "c1", Content: "", IsError: true}),
		}},
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: ContentReasoning, Reasoning: &Reasoning{Provider: "anthropic", Text: "t", Signature: "sig"}},
			{Type: ContentReasoning, Reasoning: &Reasoning{Provider: "openai", ID: "rs_1", Text: "s", Encrypted: "enc"}},
			ToolCallBlock(ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{"city": "SF",  "n":1}`)}),
			ToolCallBlock(ToolCall{ID: "c2", Name: "g", Arguments: json.RawMessage(`{"broken": `)}),
			ToolCallBlock(ToolCall{ID: "c3", Name: "h"}),
		}},
		{Role: RoleAssistant, Content: nil},
	}
	data, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	var got []Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, msgs) {
		t.Errorf("round trip mismatch\njson: %s\ngot:  %+v", data, got)
	}
}

func TestToolCallUnmarshalInlineArguments(t *testing.T) {
	var c ToolCall
	if err := json.Unmarshal([]byte(`{"id":"a","name":"f","arguments":{"x":1}}`), &c); err != nil {
		t.Fatal(err)
	}
	if string(c.Arguments) != `{"x":1}` {
		t.Errorf("arguments = %s", c.Arguments)
	}
	if err := json.Unmarshal([]byte(`{"id":"a","name":"f","arguments":null}`), &c); err != nil || c.Arguments != nil {
		t.Errorf("null arguments = %s, %v", c.Arguments, err)
	}
}
