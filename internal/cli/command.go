package cli

import "strings"

type CommandKind int

const (
	// CmdMessage is ordinary input to send to the model.
	CmdMessage CommandKind = iota
	CmdEmpty
	CmdQuit
	CmdNew
	CmdSessions
	CmdHelp
	CmdUnknown
)

type Command struct {
	Kind CommandKind
	// Text is the message for CmdMessage, or the unknown command name.
	Text string
}

// ParseCommand classifies one line of REPL input. Lines starting with "/"
// are commands; "//" escapes a message that starts with a slash.
func ParseCommand(line string) Command {
	line = strings.TrimSpace(line)
	switch {
	case line == "":
		return Command{Kind: CmdEmpty}
	case strings.HasPrefix(line, "//"):
		return Command{Kind: CmdMessage, Text: line[1:]}
	case !strings.HasPrefix(line, "/"):
		return Command{Kind: CmdMessage, Text: line}
	}
	name, _, _ := strings.Cut(line, " ")
	switch strings.ToLower(name) {
	case "/quit", "/exit", "/q":
		return Command{Kind: CmdQuit}
	case "/new":
		return Command{Kind: CmdNew}
	case "/sessions":
		return Command{Kind: CmdSessions}
	case "/help", "/?":
		return Command{Kind: CmdHelp}
	}
	return Command{Kind: CmdUnknown, Text: name}
}

const helpText = `命令：
  /sessions        列出会话
  /new             开始新会话
  /quit 或 /exit   退出
  /help            显示本帮助
以 / 开头的消息请写成 //…；生成过程中按 Ctrl+C 可中断本轮，历史会保留。`
