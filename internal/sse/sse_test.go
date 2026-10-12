package sse

import (
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"
)

func readAll(t *testing.T, input string) []Event {
	t.Helper()
	r := NewReader(strings.NewReader(input))
	var events []Event
	for {
		ev, err := r.Next()
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		events = append(events, ev)
	}
}

func TestReader(t *testing.T) {
	long := strings.Repeat("x", 200_000)
	tests := []struct {
		name  string
		input string
		want  []Event
	}{
		{
			name:  "named events",
			input: "event: a\ndata: {\"x\":1}\n\nevent: b\ndata: 2\n\n",
			want:  []Event{{Name: "a", Data: `{"x":1}`}, {Name: "b", Data: "2"}},
		},
		{
			name:  "multi-line data, comments and id",
			input: ": keepalive\nid: 7\ndata: one\ndata:two\n\n",
			want:  []Event{{ID: "7", Data: "one\ntwo"}},
		},
		{
			name:  "CRLF line endings",
			input: "event: a\r\ndata: x\r\n\r\n",
			want:  []Event{{Name: "a", Data: "x"}},
		},
		{
			name:  "trailing event without blank line",
			input: "data: last",
			want:  []Event{{Data: "last"}},
		},
		{
			name:  "extra blank lines are ignored",
			input: "\n\n\ndata: x\n\n\n",
			want:  []Event{{Data: "x"}},
		},
		{
			name:  "lines longer than bufio default",
			input: "data: " + long + "\n\n",
			want:  []Event{{Data: long}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := readAll(t, tt.input); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}
