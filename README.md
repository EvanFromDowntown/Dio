# Dio

Dio 是一个用 Go 编写的 agent，仅依赖标准库。目前包含：

- LLM 调用层：一套与提供方无关的接口，以及 OpenAI Responses API、OpenAI Chat Completions API 和 Anthropic Messages API 三个适配器。
- Agent loop：能力 tool 化、约束 hook 化、插件只负责打包。

## 目录结构

```
agent/                Agent loop、Plugin / Host 接口与可选能力接口、Decision
session/              会话持久化：Store 接口、JSONL 文件存储、内存存储、Recorder 插件、Runner
cmd/dio/              交互式命令行客户端 dio
internal/cli/         dio 的参数解析、斜杠命令、事件渲染、REPL 与单次问答
internal/webui/       dio ui：本地 HTTP/SSE API 与嵌入的单页前端（index.html）
tool/                 Tool 接口、tool.Func 适配器、Registry
plugins/readfile/     示例插件：只读 read_file 工具 + 记录日志的 hook
examples/readfile/    使用示例插件的命令行 demo
llm/                  与提供方无关的类型与接口
  client.go           Client / Stream 接口、StreamEvent、Collect
  message.go          Role、Message、ContentBlock（文本/图片/工具调用/工具结果/推理）
  tool.go             Tool、ToolCall、ToolResult、ToolChoice
  request.go          Request、Response、Usage、StopReason
  errors.go           APIError（状态码、类型、Retry-After、Retryable）
  provider.go         协议 ID（openai/chat-completions、openai/responses、anthropic/messages）与旧名兼容
  openai/
    chatcompletions/  openai/chat-completions：Chat Completions API（POST {baseURL}/chat/completions）
    responses/        openai/responses：Responses API（POST {baseURL}/responses）
  anthropic/
    messages/         anthropic/messages：Messages API（POST {baseURL}/v1/messages）
internal/
  sse/                text/event-stream 解析器
  transport/          共享的 HTTP 请求、错误解析与流式迭代器
  testutil/           适配器测试用的 httptest 辅助函数
```

## 用法

```go
import (
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/llm/anthropic/messages"
	"github.com/EvanFromDowntown/Dio/llm/openai/chatcompletions"
	"github.com/EvanFromDowntown/Dio/llm/openai/responses"
)

var client llm.Client = responses.New(
	responses.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	responses.WithBaseURL("https://my-gateway.example.com/v1"), // 可选，默认 https://api.openai.com/v1
	responses.WithHTTPClient(&http.Client{Timeout: 2 * time.Minute}),
)
// 或者：chatcompletions.New(chatcompletions.WithAPIKey(...), chatcompletions.WithBaseURL("https://my-gateway.example.com/v1"))
// 或者：messages.New(messages.WithAPIKey(...), messages.WithBaseURL("https://my-gateway.example.com"))

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
| `MessageHook` | 每条追加到历史的消息（模型响应、工具结果批次）都会经过它 |
| `FinishHook` | 运行结束时收到通知（无论成功与否） |

- hook 按插件注册顺序执行。`BeforeTool` 返回 `agent.Deny(reason)` 时，拒绝原因作为错误结果返回给模型，后续 `BeforeTool` 不再执行；`agent.ModifyArguments(args)` 会改写参数，后面的 hook 和工具都会看到改写后的参数。
- 工具返回错误、调用不存在的工具、工具 panic，都会作为错误结果回传给模型，循环继续；hook 返回 error 则中止运行。
- 运行中断时（取消、hook 出错、达到 `MaxTurns`），已请求但没有结果的工具调用会补上错误结果，`Result.Messages` 可以直接用于下一次 `Run`。

示例：

```bash
OPENAI_API_KEY=... go run ./examples/readfile -model gpt-5 "Summarize README.md"

# 兼容 Chat Completions 协议的网关，key 从指定的环境变量读取
DIO_API_KEY=... go run ./examples/readfile -provider openai/chat-completions -api-key-env DIO_API_KEY \
  -base-url https://gateway.example.com/v1 -model deepseek-flash "Summarize README.md"
```

## Session

一次对话可以跨多次运行持久化、列出和恢复：

```go
store, _ := session.NewFileStore(dir) // 默认目录用 session.DefaultDir()：~/.dio/sessions
a, _ := agent.New(ctx, agent.Config{ /* ... */ Plugins: []agent.Plugin{files, session.NewRecorder(store)}})
runner := session.Runner{Agent: a, Store: store}

sess, _ := store.Create(ctx, session.Info{Provider: "openai", Model: "gpt-5"})
runner.Run(ctx, sess.ID, llm.UserText("Remember: the code is 4729."))
runner.Run(ctx, sess.ID, llm.UserText("What was the code?")) // 自动带上完整历史
runner.Run(ctx, sess.ID)                                      // 中断后不带新输入直接续跑
```

- `Recorder` 是一个插件：通过 `MessageHook` 增量写入每条新消息，通过 `AfterModel` 写入每次响应的 usage。它写入的是 context 里携带的 session（`session.NewContext`），所以同一个 agent 可以服务多个 session。
- `Runner` 负责加载历史、先落盘新输入、补齐崩溃遗留的未完成工具调用，然后运行 agent。
- 每个 session 存成一个 `<id>.jsonl` 文件：第一行是 `session` 头（版本、ID、provider、model、metadata），后面每行是一条追加记录（`message` / `usage` / `title`）。每次追加都是一次完整的整行写入；崩溃留下的半行在读取时会被忽略，下次追加时会被截掉。

示例程序：

```bash
go run ./examples/readfile -model gpt-5 "Remember: the code is 4729."   # 默认新建，stderr 打印 session ID
go run ./examples/readfile -model gpt-5 -session <id> "What was the code?"
go run ./examples/readfile sessions                                     # 列出 session
```

## dio 客户端

```bash
go build -o dio ./cmd/dio
export DIO_API_KEY=...          # key 只从环境变量读取，变量名可用 -api-key-env 修改

# 交互模式（多轮对话都记入同一个会话，自动保存到 ~/.dio/sessions）
./dio -base-url https://gateway.example.com/v1 -model deepseek-flash -root ~/notes

# 单次问答：参数或管道（标准输入不是终端时读取全部输入作为一条消息）
./dio -model deepseek-flash "总结 README.md"
echo "总结 README.md" | ./dio -model deepseek-flash

./dio -session 20261014-101500-a1b2c3   # 恢复会话；未显式指定时沿用会话的 provider/model/base URL
./dio sessions                          # 列出会话
```

- `-provider`：`openai/chat-completions`（默认）、`openai/responses`、`anthropic/messages`。三者可以用同一个 `.../v1` 形式的 base URL，`anthropic/messages` 会自动去掉末尾的 `/v1`。旧名 `chatcompletions`、`openai`、`anthropic` 仍然可用（参数、`DIO_PROVIDER`、Web 设置、旧会话文件），会自动换成新 ID；新写入的会话一律记录新 ID。
- 环境变量 `DIO_PROVIDER`、`DIO_MODEL`、`DIO_BASE_URL` 可以作为默认值，命令行参数优先。
- 其他参数：`-root`（read_file 的只读工作目录，默认当前目录）、`-max-turns`、`-max-tokens`、`-reasoning-effort`、`-temperature`、`-show-reasoning`、`-sessions-dir`。
- 对话中可用的命令：`/sessions`、`/new`、`/quit`（或 `/exit`）、`/help`。生成过程中按 Ctrl+C 只中断当前这一轮，历史会保留；在提示符处按 Ctrl+C 退出。
- 回答输出到 stdout；工具调用以 `[工具] read_file(path=notes.txt)`、`[结果] …` / `[出错] …` 的形式显示，单次模式下这些状态行输出到 stderr。

## Web 界面

```bash
export DIO_API_KEY=...   # 可选：也可以在界面“设置”里填写
./dio ui -base-url https://gateway.example.com/v1 -model deepseek-flash -root ~/notes
# 打开 http://127.0.0.1:8080/
```

- 左侧是会话列表（标题、更新时间、消息数、token 用量、ID），可以新建会话，也可以点选已有会话查看历史并继续聊；会话仍保存在 `~/.dio/sessions`，和 CLI 共用。
- 回答逐字流式显示，推理过程折叠显示，工具调用和结果以 `工具 read_file(path=…)`、`结果 … 字节 · 预览` / `出错 …` 的形式显示；生成中可以点“停止”，已生成的内容会保留。
- “设置”里可以修改协议、模型、Base URL、API key、工作目录、最大轮数、最大 tokens、推理强度和 temperature。启动参数和 `DIO_*` 环境变量作为初始值。
- API key 只保存在 dio 进程内存中，不写入磁盘，接口也不会返回它。
- 只能监听本机地址（127.0.0.1、::1、localhost）；服务端会拒绝 Host 不是本机的请求（防 DNS rebinding）和跨站的写请求。

HTTP API（前端使用的同一套）：

| 方法与路径 | 作用 |
| --- | --- |
| `GET /api/config`、`PUT /api/config` | 读取 / 修改设置（`api_key` 非空时替换，`clear_api_key` 清除） |
| `GET /api/sessions`、`POST /api/sessions` | 列出 / 新建会话 |
| `GET /api/sessions/{id}` | 会话详情与可展示的历史 |
| `POST /api/sessions/{id}/messages` | 发送消息，以 SSE 返回 `text`、`reasoning`、`item`（工具行）、`message_end`、`done` 事件 |
| `POST /api/sessions/{id}/cancel` | 停止正在生成的回答 |

写请求需要 `Content-Type: application/json`。

## 开发

```bash
go vet ./...
go test ./...
```
