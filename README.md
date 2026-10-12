# Dio

Dio 是一个用 Go 编写的 agent，仅依赖标准库。目前包含：

- LLM 调用层：一套与提供方无关的接口，以及 OpenAI Responses API、OpenAI Chat Completions API 和 Anthropic Messages API 三个适配器。
- Agent loop：能力 tool 化、约束 hook 化、插件只负责打包。

## 目录结构

```
agent/                Agent loop、Plugin / Host 接口与可选能力接口、Decision
tool/                 Tool 接口、tool.Func 适配器、Registry
plugins/readfile/     示例插件：只读 read_file 工具 + 记录日志的 hook
examples/readfile/    使用示例插件的命令行 demo
llm/                  与提供方无关的类型与接口
  client.go           Client / Stream 接口、StreamEvent、Collect
  message.go          Role、Message、ContentBlock（文本/图片/工具调用/工具结果/推理）
  tool.go             Tool、ToolCall、ToolResult、ToolChoice
  request.go          Request、Response、Usage、StopReason
  errors.go           APIError（状态码、类型、Retry-After、Retryable）
  openai/             OpenAI Responses API 适配器（POST {baseURL}/responses）
  chatcompletions/    OpenAI Chat Completions API 适配器（POST {baseURL}/chat/completions）
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
- `Request.Extra` 会合并到请求体顶层，用于传递提供方或网关的专有参数（如 OpenAI 的 `store`、`include`，Chat Completions 的 `response_format`）。
- Chat Completions 适配器：`MaxTokens` 默认以 `max_completion_tokens` 发送（可用 `WithMaxTokensField("max_tokens")` 改）；`Reasoning.Effort` 映射为 `reasoning_effort`；推理内容读写 `reasoning_content`（DeepSeek 等模型的约定，可用 `WithReasoningReplay(false)` 关闭回放）；流式默认带 `stream_options.include_usage`。该协议没有工具错误标记，`IsError` 的工具结果会加上 `Error: ` 前缀。
- 推理模型的推理 token 也计入输出上限，上限太小时可能只有推理、没有正文（`StopMaxTokens`）。
- 错误统一为 `*llm.APIError`，可以用 `llm.IsRetryable(err)` 判断是否值得重试（429、5xx、overloaded 等）。

## Agent loop

一般不需要手写上面的循环，用 `agent` 包即可：

```go
files := readfile.New(".")
defer files.Close()

a, err := agent.New(ctx, agent.Config{
	Client:   client,
	Model:    "gpt-5",
	System:   "You are a concise assistant.",
	MaxTurns: 10,
	Plugins:  []agent.Plugin{files},
})

// 非流式
res, err := a.Run(ctx, []llm.Message{llm.UserText("Summarize README.md")})
fmt.Println(res.Text(), res.FinishReason, res.Usage)

// 流式：最后一个事件总是 EventDone，err 为本次运行的错误
for ev, err := range a.Stream(ctx, msgs) {
	if ev.Type == agent.EventModel && ev.LLM.Type == llm.EventTextDelta {
		fmt.Print(ev.LLM.Delta)
	}
}
```

每一轮的流程：组装 system prompt（`Config.System` 加上各插件的 `SystemPrompt` 片段）→ `BeforeModel` → 调用模型 → `AfterModel` → 没有工具调用就结束；否则对每个调用依次执行 `BeforeTool` → 执行工具 → `AfterTool`，把结果追加到历史后进入下一轮。

插件实现 `Plugin`（`Name`、`Init`），再按需实现可选能力接口，agent 通过类型断言发现它们：

| 接口 | 作用 |
| --- | --- |
| `ToolProvider` | 提供工具 |
| `PromptProvider` | 提供 system prompt 片段 |
| `ModelHook` | 修改请求、修改响应 |
| `ToolHook` | 放行、拒绝或改写工具调用，改写工具结果 |
| `FinishHook` | 运行结束时收到通知（无论成功与否） |

- hook 按插件注册顺序执行。`BeforeTool` 返回 `agent.Deny(reason)` 时，拒绝原因作为错误结果返回给模型，后续 `BeforeTool` 不再执行；`agent.ModifyArguments(args)` 会改写参数，后面的 hook 和工具都会看到改写后的参数。
- 工具返回错误、调用不存在的工具、工具 panic，都会作为错误结果回传给模型，循环继续；hook 返回 error 则中止运行。
- 运行中断时（取消、hook 出错、达到 `MaxTurns`），已请求但没有结果的工具调用会补上错误结果，`Result.Messages` 可以直接用于下一次 `Run`。

示例：

```bash
OPENAI_API_KEY=... go run ./examples/readfile -model gpt-5 "Summarize README.md"

# 兼容 Chat Completions 协议的网关，key 从指定的环境变量读取
DIO_API_KEY=... go run ./examples/readfile -provider chatcompletions -api-key-env DIO_API_KEY \
  -base-url https://gateway.example.com/v1 -model deepseek-flash "Summarize README.md"
```

## 开发

```bash
go vet ./...
go test ./...
```
