# Dio

Dio 是一个用 Go 编写的 agent。当前已完成 LLM 调用层：一套与提供方无关的接口，以及 OpenAI Responses API 和 Anthropic Messages API 两个适配器。仅依赖标准库。

## 目录结构

```
llm/                  与提供方无关的类型与接口
  client.go           Client / Stream 接口、StreamEvent、Collect
  message.go          Role、Message、ContentBlock（文本/图片/工具调用/工具结果/推理）
  tool.go             Tool、ToolCall、ToolResult、ToolChoice
  request.go          Request、Response、Usage、StopReason
  errors.go           APIError（状态码、类型、Retry-After、Retryable）
  openai/             OpenAI Responses API 适配器（POST {baseURL}/responses）
  anthropic/          Anthropic Messages API 适配器（POST {baseURL}/v1/messages）
internal/
  sse/                text/event-stream 解析器
  transport/          共享的 HTTP 请求、错误解析与流式迭代器
  testutil/           适配器测试用的 httptest 辅助函数
```

## 用法

```go
import (
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/llm/anthropic"
	"github.com/EvanFromDowntown/Dio/llm/openai"
)

var client llm.Client = openai.New(
	openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	openai.WithBaseURL("https://my-gateway.example.com/v1"), // 可选，默认 https://api.openai.com/v1
	openai.WithHTTPClient(&http.Client{Timeout: 2 * time.Minute}),
)
// 或者：anthropic.New(anthropic.WithAPIKey(...), anthropic.WithBaseURL("https://my-gateway.example.com"))

req := &llm.Request{
	Model:     "gpt-5",
	System:    "You are a helpful assistant.",
	Messages:  []llm.Message{llm.UserText("What's the weather in SF?")},
	Tools:     []llm.Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}},
	MaxTokens: 1024,
}

// 非流式
resp, err := client.Generate(ctx, req)

// 流式
s, err := client.Stream(ctx, req)
if err != nil { /* HTTP 层错误 */ }
defer s.Close()
for s.Next() {
	ev := s.Event()
	if ev.Type == llm.EventTextDelta {
		fmt.Print(ev.Delta)
	}
	// 最后一个事件是 EventDone，ev.Response 为完整响应
}
if err := s.Err(); err != nil { /* 流中途的错误 */ }
// 只要最终结果时：resp, err := llm.Collect(s)
```

### 工具调用循环

```go
for {
	resp, err := client.Generate(ctx, req)
	if err != nil {
		return err
	}
	req.Messages = append(req.Messages, resp.Message())
	if resp.StopReason != llm.StopToolUse {
		break
	}
	var results []llm.ToolResult
	for _, call := range resp.ToolCalls() {
		out, err := runTool(call.Name, call.Arguments)
		results = append(results, llm.ToolResult{ToolCallID: call.ID, Content: out, IsError: err != nil})
	}
	req.Messages = append(req.Messages, llm.ToolResults(results...))
}
```

### 约定

- `WithBaseURL` 沿用各官方 SDK 的习惯：OpenAI 的 base URL 包含 `/v1`，Anthropic 的不包含。
- Anthropic 适配器支持 `WithAPIKey`（`x-api-key`）和 `WithAuthToken`（`Authorization: Bearer`，部分网关需要）；`WithHeader` 可追加任意请求头（如 `anthropic-beta`）。
- `Request.Extra` 会合并到请求体顶层，用于传递提供方或网关的专有参数（如 OpenAI 的 `store`、`include`）。
- 错误统一为 `*llm.APIError`，可以用 `llm.IsRetryable(err)` 判断是否值得重试（429、5xx、overloaded 等）。

## 开发

```bash
go vet ./...
go test ./...
```
