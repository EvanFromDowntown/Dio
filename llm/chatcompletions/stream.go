package chatcompletions

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/EvanFromDowntown/Dio/internal/sse"
	"github.com/EvanFromDowntown/Dio/internal/transport"
	"github.com/EvanFromDowntown/Dio/llm"
)

// streamDecoder assembles chat.completion.chunk events. Reasoning, text and
// each tool call (keyed by its index) become blocks in arrival order; all
// blocks are closed when the stream finishes, because chunks may interleave
// them.
type streamDecoder struct {
	started      bool
	id, model    string
	blocks       []*partial
	reasoningIdx int
	textIdx      int
	toolIdx      map[int]int
	lastTool     int
	finishReason string
	finished     bool
	usage        *apiUsage
}

type partial struct {
	typ      llm.ContentType
	text     strings.Builder
	id, name string
	args     strings.Builder
}

func (d *streamDecoder) decode(ev sse.Event) ([]llm.StreamEvent, bool, error) {
	data := strings.TrimSpace(ev.Data)
	if data == "" {
		return nil, false, nil
	}
	if data == "[DONE]" {
		events, err := d.finish()
		return events, true, err
	}
	var chunk apiResponse
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return nil, false, fmt.Errorf("chatcompletions: decode stream chunk: %w", err)
	}
	if chunk.Error != nil {
		return nil, false, transport.ErrorFromBody(ProviderName, []byte(data))
	}

	var events []llm.StreamEvent
	if !d.started {
		d.started = true
		d.reasoningIdx, d.textIdx, d.lastTool = -1, -1, -1
		d.toolIdx = map[int]int{}
		d.id, d.model = chunk.ID, chunk.Model
		events = append(events, llm.StreamEvent{Type: llm.EventStart, Response: &llm.Response{ID: chunk.ID, Model: chunk.Model}})
	}
	if chunk.Usage != nil {
		d.usage = chunk.Usage
	}
	for _, c := range chunk.Choices {
		if c.Index != 0 {
			continue
		}
		if c.FinishReason != "" {
			d.finishReason = c.FinishReason
		}
		if c.Delta == nil {
			continue
		}
		if s := c.Delta.ReasoningContent; s != "" {
			events = append(events, d.appendText(&d.reasoningIdx, llm.ContentReasoning, s, llm.EventReasoningDelta)...)
		}
		if s := contentText(c.Delta.Content); s != "" {
			events = append(events, d.appendText(&d.textIdx, llm.ContentText, s, llm.EventTextDelta)...)
		}
		for _, tc := range c.Delta.ToolCalls {
			events = append(events, d.appendToolCall(tc)...)
		}
	}
	return events, false, nil
}

func (d *streamDecoder) open(p *partial, start llm.ContentBlock) (int, llm.StreamEvent) {
	idx := len(d.blocks)
	d.blocks = append(d.blocks, p)
	return idx, llm.StreamEvent{Type: llm.EventContentStart, Index: idx, Block: &start}
}

func (d *streamDecoder) appendText(idx *int, typ llm.ContentType, delta string, evType llm.EventType) []llm.StreamEvent {
	var events []llm.StreamEvent
	if *idx < 0 {
		start := llm.TextBlock("")
		if typ == llm.ContentReasoning {
			start = reasoningBlock("")
		}
		var ev llm.StreamEvent
		*idx, ev = d.open(&partial{typ: typ}, start)
		events = append(events, ev)
	}
	d.blocks[*idx].text.WriteString(delta)
	return append(events, llm.StreamEvent{Type: evType, Index: *idx, Delta: delta})
}

// appendToolCall keys fragments by index. Servers that omit the index get
// a new call whenever a fragment carries a new id.
func (d *streamDecoder) appendToolCall(tc apiToolCall) []llm.StreamEvent {
	key := d.lastTool
	switch {
	case tc.Index != nil:
		key = *tc.Index
	case tc.ID != "" || key < 0:
		key = len(d.toolIdx)
	}
	d.lastTool = key

	var events []llm.StreamEvent
	idx, ok := d.toolIdx[key]
	if !ok {
		p := &partial{typ: llm.ContentToolCall, id: tc.ID, name: tc.Function.Name}
		var ev llm.StreamEvent
		idx, ev = d.open(p, llm.ToolCallBlock(llm.ToolCall{ID: tc.ID, Name: tc.Function.Name}))
		d.toolIdx[key] = idx
		events = append(events, ev)
	} else {
		p := d.blocks[idx]
		if p.id == "" {
			p.id = tc.ID
		}
		if p.name == "" {
			p.name = tc.Function.Name
		}
	}
	if args := tc.Function.Arguments; args != "" {
		d.blocks[idx].args.WriteString(args)
		events = append(events, llm.StreamEvent{Type: llm.EventToolCallDelta, Index: idx, Delta: args})
	}
	return events
}

func (d *streamDecoder) finish() ([]llm.StreamEvent, error) {
	if d.finished {
		return nil, nil
	}
	d.finished = true
	resp := &llm.Response{
		ID: d.id, Model: d.model, Content: make([]llm.ContentBlock, 0, len(d.blocks)),
		StopReason: convertFinishReason(d.finishReason), ProviderStopReason: d.finishReason,
	}
	if d.usage != nil {
		resp.Usage = convertUsage(d.usage)
	}
	var events []llm.StreamEvent
	for i, p := range d.blocks {
		var block llm.ContentBlock
		switch p.typ {
		case llm.ContentReasoning:
			block = reasoningBlock(p.text.String())
		case llm.ContentText:
			block = llm.TextBlock(p.text.String())
		case llm.ContentToolCall:
			block = toolCallBlock(p.id, p.name, p.args.String())
		}
		resp.Content = append(resp.Content, block)
		events = append(events, llm.StreamEvent{Type: llm.EventContentStop, Index: i, Block: &block})
	}
	return append(events, llm.StreamEvent{Type: llm.EventDone, Response: resp}), nil
}

// eof completes streams from servers that end after the finish_reason chunk
// without sending [DONE].
func (d *streamDecoder) eof() ([]llm.StreamEvent, bool, error) {
	if d.finishReason == "" {
		return nil, false, nil
	}
	events, err := d.finish()
	return events, true, err
}
