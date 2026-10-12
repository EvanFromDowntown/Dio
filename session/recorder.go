package session

import (
	"context"
	"fmt"
	"iter"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
)

// Recorder is an agent plugin that persists a run incrementally: every
// message the run appends (via MessageHook) and the usage of every model
// response (via AfterModel). It writes to the session named by the run's
// context (see NewContext) and does nothing for runs without one, so one
// agent can serve many sessions.
type Recorder struct {
	store Store
}

var (
	_ agent.ModelHook   = (*Recorder)(nil)
	_ agent.MessageHook = (*Recorder)(nil)
)

func NewRecorder(store Store) *Recorder { return &Recorder{store: store} }

func (r *Recorder) Name() string { return "session" }

func (r *Recorder) Init(context.Context, agent.Host) error { return nil }

func (r *Recorder) BeforeModel(context.Context, *llm.Request) error { return nil }

func (r *Recorder) AfterModel(ctx context.Context, resp *llm.Response) error {
	id, ok := IDFromContext(ctx)
	if !ok {
		return nil
	}
	return r.store.AddUsage(context.WithoutCancel(ctx), id, resp.Usage)
}

func (r *Recorder) OnMessage(ctx context.Context, msg llm.Message) error {
	id, ok := IDFromContext(ctx)
	if !ok {
		return nil
	}
	return r.store.AppendMessages(ctx, id, msg)
}

// Runner continues sessions with an agent. The agent must include a Recorder
// for the same Store, otherwise the run's output is not persisted.
type Runner struct {
	Agent *agent.Agent
	Store Store
}

// Run loads session id, persists input, and runs the agent on the full
// history. With no input it resumes an interrupted session as is.
func (r Runner) Run(ctx context.Context, id string, input ...llm.Message) (*agent.Result, error) {
	ctx, history, err := r.prepare(ctx, id, input)
	if err != nil {
		return nil, err
	}
	return r.Agent.Run(ctx, history)
}

// Stream is the streaming form of Run.
func (r Runner) Stream(ctx context.Context, id string, input ...llm.Message) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		ctx, history, err := r.prepare(ctx, id, input)
		if err != nil {
			yield(agent.Event{Type: agent.EventDone}, err)
			return
		}
		for ev, err := range r.Agent.Stream(ctx, history) {
			if !yield(ev, err) {
				return
			}
		}
	}
}

func (r Runner) prepare(ctx context.Context, id string, input []llm.Message) (context.Context, []llm.Message, error) {
	sess, err := r.Store.Get(ctx, id)
	if err != nil {
		return nil, nil, err
	}
	var added []llm.Message
	if pending := PendingToolCalls(sess.Messages); len(pending) > 0 {
		results := make([]llm.ToolResult, len(pending))
		for i, c := range pending {
			results[i] = llm.ToolResult{
				ToolCallID: c.ID, IsError: true,
				Content: "Tool call was interrupted before a result was recorded.",
			}
		}
		added = append(added, llm.ToolResults(results...))
	}
	added = append(added, input...)
	if err := r.Store.AppendMessages(ctx, id, added...); err != nil {
		return nil, nil, fmt.Errorf("session: persist input: %w", err)
	}
	if sess.Title == "" {
		if title := Title(input); title != "" {
			if err := r.Store.SetTitle(ctx, id, title); err != nil {
				return nil, nil, err
			}
		}
	}
	return NewContext(ctx, id), append(sess.Messages, added...), nil
}
