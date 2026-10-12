package openai

import (
	"encoding/json"
	"fmt"

	"github.com/EvanFromDowntown/Dio/internal/sse"
	"github.com/EvanFromDowntown/Dio/internal/transport"
	"github.com/EvanFromDowntown/Dio/llm"
)

// streamDecoder maps Responses API events to unified events. Text parts are
// keyed by (output_index, content_index); tool calls and reasoning items by
// output_index. Unified indices are assigned in arrival order, which matches
// the order convertResponse produces for the final response.
type streamDecoder struct {
	next      int
	textParts map[[2]int]int
	items     map[int]int
}

func newStreamDecoder() *streamDecoder {
	return &streamDecoder{textParts: map[[2]int]int{}, items: map[int]int{}}
}

type streamPayload struct {
	Type         string         `json:"type"`
	OutputIndex  int            `json:"output_index"`
	ContentIndex int            `json:"content_index"`
	SummaryIndex int            `json:"summary_index"`
	Delta        string         `json:"delta"`
	Item         *outputItem    `json:"item"`
	Part         *outputContent `json:"part"`
	Response     *apiResponse   `json:"response"`
}

func (d *streamDecoder) decode(ev sse.Event) ([]llm.StreamEvent, bool, error) {
	if ev.Data == "" || ev.Data == "[DONE]" {
		return nil, false, nil
	}
	var p streamPayload
	if err := json.Unmarshal([]byte(ev.Data), &p); err != nil {
		return nil, false, fmt.Errorf("openai: decode stream event: %w", err)
	}
	switch p.Type {
	case "response.created":
		if p.Response == nil {
			return nil, false, nil
		}
		return []llm.StreamEvent{{
			Type:     llm.EventStart,
			Response: &llm.Response{ID: p.Response.ID, Model: p.Response.Model},
		}}, false, nil

	case "response.output_item.added":
		if p.Item == nil {
			return nil, false, nil
		}
		var block llm.ContentBlock
		switch p.Item.Type {
		case "function_call":
			block = llm.ToolCallBlock(llm.ToolCall{ID: p.Item.CallID, Name: p.Item.Name})
		case "reasoning":
			block = llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, ID: p.Item.ID}}
		default:
			return nil, false, nil
		}
		idx := d.alloc()
		d.items[p.OutputIndex] = idx
		return []llm.StreamEvent{{Type: llm.EventContentStart, Index: idx, Block: &block}}, false, nil

	case "response.content_part.added":
		if p.Part == nil || (p.Part.Type != "output_text" && p.Part.Type != "refusal") {
			return nil, false, nil
		}
		idx := d.alloc()
		d.textParts[[2]int{p.OutputIndex, p.ContentIndex}] = idx
		block := llm.TextBlock("")
		return []llm.StreamEvent{{Type: llm.EventContentStart, Index: idx, Block: &block}}, false, nil

	case "response.output_text.delta", "response.refusal.delta":
		idx, ok := d.textParts[[2]int{p.OutputIndex, p.ContentIndex}]
		if !ok {
			return nil, false, nil
		}
		return []llm.StreamEvent{{Type: llm.EventTextDelta, Index: idx, Delta: p.Delta}}, false, nil

	case "response.content_part.done":
		idx, ok := d.textParts[[2]int{p.OutputIndex, p.ContentIndex}]
		if !ok || p.Part == nil {
			return nil, false, nil
		}
		text := p.Part.Text
		if p.Part.Type == "refusal" {
			text = p.Part.Refusal
		}
		block := llm.TextBlock(text)
		return []llm.StreamEvent{{Type: llm.EventContentStop, Index: idx, Block: &block}}, false, nil

	case "response.function_call_arguments.delta":
		idx, ok := d.items[p.OutputIndex]
		if !ok {
			return nil, false, nil
		}
		return []llm.StreamEvent{{Type: llm.EventToolCallDelta, Index: idx, Delta: p.Delta}}, false, nil

	case "response.reasoning_summary_part.added":
		idx, ok := d.items[p.OutputIndex]
		if !ok || p.SummaryIndex == 0 {
			return nil, false, nil
		}
		return []llm.StreamEvent{{Type: llm.EventReasoningDelta, Index: idx, Delta: "\n\n"}}, false, nil

	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		idx, ok := d.items[p.OutputIndex]
		if !ok {
			return nil, false, nil
		}
		return []llm.StreamEvent{{Type: llm.EventReasoningDelta, Index: idx, Delta: p.Delta}}, false, nil

	case "response.output_item.done":
		idx, ok := d.items[p.OutputIndex]
		if !ok || p.Item == nil {
			return nil, false, nil
		}
		blocks := convertOutputItem(*p.Item)
		if len(blocks) != 1 {
			return nil, false, nil
		}
		return []llm.StreamEvent{{Type: llm.EventContentStop, Index: idx, Block: &blocks[0]}}, false, nil

	case "response.completed", "response.incomplete", "response.failed":
		if p.Response == nil {
			return nil, false, fmt.Errorf("openai: %s event without response", p.Type)
		}
		resp, err := convertResponse(p.Response)
		if err != nil {
			return nil, false, err
		}
		return []llm.StreamEvent{{Type: llm.EventDone, Response: resp}}, true, nil

	case "error":
		return nil, false, transport.ErrorFromBody(ProviderName, []byte(ev.Data))
	}
	return nil, false, nil
}

func (d *streamDecoder) alloc() int {
	idx := d.next
	d.next++
	return idx
}
