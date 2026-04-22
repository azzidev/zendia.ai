package datasource

import (
	"regexp"
	"strings"
)

var (
	// remove tags HTML
	reHTML = regexp.MustCompile(`<[^>]+>`)
	// remove templates {{ }}
	reTemplate = regexp.MustCompile(`\{\{[^}]*\}\}`)
	// remove [[Arquivo:...]] e [[File:...]]
	reFile = regexp.MustCompile(`\[\[(Arquivo|File|Imagem|Image):[^\]]*\]\]`)
	// converte [[link|texto]] → texto
	reLinkText = regexp.MustCompile(`\[\[[^|\]]*\|([^\]]*)\]\]`)
	// converte [[link]] → link
	reLink = regexp.MustCompile(`\[\[([^\]]*)\]\]`)
	// remove refs <ref>...</ref>
	reRef = regexp.MustCompile(`(?s)<ref[^>]*>.*?</ref>`)
	reRefSelf = regexp.MustCompile(`<ref[^/]*/\s*>`)
	// remove categorias
	reCategory = regexp.MustCompile(`\[\[(Categoria|Category):[^\]]*\]\]`)
	// remove tabelas {| ... |}
	reTable = regexp.MustCompile(`(?s)\{\|.*?\|\}`)
	// remove headers == ... ==
	reHeader = regexp.MustCompile(`={2,}([^=]+)={2,}`)
	// remove múltiplos espaços/newlines
	reSpaces   = regexp.MustCompile(`[ \t]+`)
	reNewlines = regexp.MustCompile(`\n{3,}`)
	// remove marcadores wiki
	reBold   = regexp.MustCompile(`'{2,3}`)
	reComment = regexp.MustCompile(`(?s)<!--.*?-->`)
)

func CleanWikiText(raw string) string {
	text := raw

	text = reComment.ReplaceAllString(text, "")
	text = reRef.ReplaceAllString(text, "")
	text = reRefSelf.ReplaceAllString(text, "")
	text = reTable.ReplaceAllString(text, "")
	text = reTemplate.ReplaceAllString(text, "")
	text = reFile.ReplaceAllString(text, "")
	text = reCategory.ReplaceAllString(text, "")
	text = reHeader.ReplaceAllString(text, "$1")
	text = reLinkText.ReplaceAllString(text, "$1")
	text = reLink.ReplaceAllString(text, "$1")
	text = reHTML.ReplaceAllString(text, "")
	text = reBold.ReplaceAllString(text, "")

	// limpa lixo restante
	text = strings.ReplaceAll(text, "&nbsp;", " ")
	text = strings.ReplaceAll(text, "&amp;", "&")
	text = strings.ReplaceAll(text, "&lt;", "<")
	text = strings.ReplaceAll(text, "&gt;", ">")
	text = strings.ReplaceAll(text, "&quot;", "\"")

	text = reSpaces.ReplaceAllString(text, " ")
	text = reNewlines.ReplaceAllString(text, "\n\n")

	// remove linhas que são só pontuação/espaço
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if len(line) > 20 { // só mantém linhas com conteúdo real
			lines = append(lines, line)
		}
	}

	return strings.Join(lines, "\n")
}
