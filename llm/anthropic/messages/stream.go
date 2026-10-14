package messages

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/EvanFromDowntown/Dio/internal/sse"
	"github.com/EvanFromDowntown/Dio/internal/transport"
	"github.com/EvanFromDowntown/Dio/llm"
)

// streamDecoder assembles Messages API events into a response. Provider block
// indices are remapped so unsupported block types leave no gaps.
type streamDecoder struct {
	resp    apiResponse
	blocks  []*partialBlock
	indexOf map[int]int
}

type partialBlock struct {
	block apiBlock
	text  strings.Builder
	json  strings.Builder
	done  bool
}

func newStreamDecoder() *streamDecoder {
	return &streamDecoder{indexOf: map[int]int{}}
}

type streamPayload struct {
	Type         string       `json:"type"`
	Index        int          `json:"index"`
	Message      *apiResponse `json:"message"`
	ContentBlock *apiBlock    `json:"content_block"`
	Delta        struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage *struct {
		InputTokens              *int `json:"input_tokens"`
		OutputTokens             *int `json:"output_tokens"`
		CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
	} `json:"usage"`
}

func (d *streamDecoder) decode(ev sse.Event) ([]llm.StreamEvent, bool, error) {
	if ev.Data == "" {
		return nil, false, nil
	}
	var p streamPayload
	if err := json.Unmarshal([]byte(ev.Data), &p); err != nil {
		return nil, false, fmt.Errorf("anthropic: decode stream event: %w", err)
	}
	switch p.Type {
	case "message_start":
		if p.Message == nil {
			return nil, false, nil
		}
		d.resp = *p.Message
		d.resp.Content = nil
		return []llm.StreamEvent{{
			Type:     llm.EventStart,
			Response: &llm.Response{ID: d.resp.ID, Model: d.resp.Model},
		}}, false, nil

	case "content_block_start":
		if p.ContentBlock == nil {
			return nil, false, nil
		}
		start, ok := convertBlock(*p.ContentBlock)
		if !ok {
			return nil, false, nil
		}
		if start.ToolCall != nil {
			start.ToolCall.Arguments = nil
		}
		idx := len(d.blocks)
		d.indexOf[p.Index] = idx
		pb := &partialBlock{block: *p.ContentBlock}
		pb.text.WriteString(p.ContentBlock.Text + p.ContentBlock.Thinking)
		d.blocks = append(d.blocks, pb)
		return []llm.StreamEvent{{Type: llm.EventContentStart, Index: idx, Block: &start}}, false, nil

	case "content_block_delta":
		idx, ok := d.indexOf[p.Index]
		if !ok {
			return nil, false, nil
		}
		pb := d.blocks[idx]
		switch p.Delta.Type {
		case "text_delta":
			pb.text.WriteString(p.Delta.Text)
			return []llm.StreamEvent{{Type: llm.EventTextDelta, Index: idx, Delta: p.Delta.Text}}, false, nil
		case "thinking_delta":
			pb.text.WriteString(p.Delta.Thinking)
			return []llm.StreamEvent{{Type: llm.EventReasoningDelta, Index: idx, Delta: p.Delta.Thinking}}, false, nil
		case "input_json_delta":
			pb.json.WriteString(p.Delta.PartialJSON)
			return []llm.StreamEvent{{Type: llm.EventToolCallDelta, Index: idx, Delta: p.Delta.PartialJSON}}, false, nil
		case "signature_delta":
			pb.block.Signature += p.Delta.Signature
		}
		return nil, false, nil

	case "content_block_stop":
		idx, ok := d.indexOf[p.Index]
		if !ok {
			return nil, false, nil
		}
		block := d.finish(d.blocks[idx])
		return []llm.StreamEvent{{Type: llm.EventContentStop, Index: idx, Block: &block}}, false, nil

	case "message_delta":
		if p.Delta.StopReason != "" {
			d.resp.StopReason = p.Delta.StopReason
		}
		if u := p.Usage; u != nil {
			setIf(&d.resp.Usage.InputTokens, u.InputTokens)
			setIf(&d.resp.Usage.OutputTokens, u.OutputTokens)
			setIf(&d.resp.Usage.CacheCreationInputTokens, u.CacheCreationInputTokens)
			setIf(&d.resp.Usage.CacheReadInputTokens, u.CacheReadInputTokens)
		}
		return nil, false, nil

	case "message_stop":
		resp := convertResponse(&d.resp)
		for _, pb := range d.blocks {
			resp.Content = append(resp.Content, d.finish(pb))
		}
		return []llm.StreamEvent{{Type: llm.EventDone, Response: resp}}, true, nil

	case "error":
		return nil, false, transport.ErrorFromBody(ProviderName, []byte(ev.Data))
	}
	return nil, false, nil
}

func (d *streamDecoder) finish(pb *partialBlock) llm.ContentBlock {
	if !pb.done {
		pb.done = true
		switch pb.block.Type {
		case "text":
			pb.block.Text = pb.text.String()
		case "thinking":
			pb.block.Thinking = pb.text.String()
		case "tool_use":
			pb.block.Input = json.RawMessage(`{}`)
			if pb.json.Len() > 0 {
				pb.block.Input = json.RawMessage(pb.json.String())
			}
		}
	}
	block, _ := convertBlock(pb.block)
	return block
}

func setIf(dst *int, v *int) {
	if v != nil {
		*dst = *v
	}
}
