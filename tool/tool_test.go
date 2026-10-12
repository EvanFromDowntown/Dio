package tool

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/EvanFromDowntown/Dio/llm"
)

func fn(name string) *Func {
	return &Func{
		ToolName: name, ToolDescription: name + " desc",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn: func(_ context.Context, args json.RawMessage) (string, error) {
			return name + string(args), nil
		},
	}
}

func TestRegistry(t *testing.T) {
	r := NewRegistry()
	if r.LLMTools() != nil {
		t.Error("empty registry should declare no tools")
	}
	if err := r.Register(fn("b"), fn("a")); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []Tool{fn("a"), fn(""), nil} {
		if err := r.Register(bad); err == nil {
			t.Errorf("Register(%v) should fail", bad)
		}
	}
	if r.Len() != 2 {
		t.Errorf("Len = %d", r.Len())
	}
	got, ok := r.Get("a")
	if !ok {
		t.Fatal("a not found")
	}
	if out, _ := got.Execute(context.Background(), json.RawMessage(`{}`)); out != "a{}" {
		t.Errorf("Execute = %q", out)
	}
	if _, ok := r.Get("missing"); ok {
		t.Error("missing tool found")
	}

	want := []llm.Tool{
		{Name: "b", Description: "b desc", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{Name: "a", Description: "a desc", InputSchema: json.RawMessage(`{"type":"object"}`)},
	}
	if !reflect.DeepEqual(r.LLMTools(), want) {
		t.Errorf("LLMTools = %+v", r.LLMTools())
	}

	tools := r.Tools()
	tools[0] = nil
	if r.Tools()[0] == nil {
		t.Error("Tools exposes internal slice")
	}
}

func TestRegisterStopsAtFirstError(t *testing.T) {
	r := NewRegistry()
	err := r.Register(fn("x"), fn("x"), fn("y"))
	if err == nil || !strings.Contains(err.Error(), `"x"`) {
		t.Errorf("err = %v", err)
	}
	if _, ok := r.Get("y"); ok {
		t.Error("tools after the failing one should not be registered")
	}
}
