import type { ReactNode } from "react";

// A small Markdown subset for chat bubbles: headings, paragraphs, ordered and
// unordered lists, code fences, inline code, bold and italics. Everything is
// emitted as React elements and text nodes (never raw HTML), so any markup in
// the text renders as inert text. Unfinished markup (a lone ** or an open
// fence while a reply is still streaming) renders literally or as an open block.

export type RenderCitation = (index: number, key: string) => ReactNode;

// `code` | **bold** | *italic* (not inside words, so 2*3*4 stays literal) | [N]
const INLINE =
  /`([^`\n]+)`|\*\*(\S(?:[^\n]*?\S)?)\*\*|(^|[^\w*])\*([^\s*](?:[^*\n]*[^\s*])?)\*(?![\w*])|\[(\d+)\]/;

function renderInline(text: string, renderCitation: RenderCitation, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = new RegExp(INLINE.source, "g");
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    const [whole, code, bold, italicLead, italic, citation] = match;
    const key = `${keyPrefix}.${match.index}`;
    let start = match.index;
    if (italic !== undefined) start += italicLead.length;
    if (start > last) out.push(text.slice(last, start));
    if (code !== undefined) {
      out.push(<code key={key}>{code}</code>);
    } else if (bold !== undefined) {
      out.push(<strong key={key}>{renderInline(bold, renderCitation, key)}</strong>);
    } else if (italic !== undefined) {
      out.push(<em key={key}>{renderInline(italic, renderCitation, key)}</em>);
    } else {
      out.push(renderCitation(parseInt(citation, 10), key));
    }
    last = match.index + whole.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function renderLines(lines: string[], renderCitation: RenderCitation, keyPrefix: string): ReactNode[] {
  return lines.flatMap((line, i) => {
    const inline = renderInline(line, renderCitation, `${keyPrefix}.${i}`);
    return i === 0 ? inline : [<br key={`${keyPrefix}.br${i}`} />, ...inline];
  });
}

const FENCE = /^\s*```/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const NUMBERED = /^\s*(\d+)[.)]\s+(.*)$/;

export function renderMarkdown(text: string, renderCitation: RenderCitation): ReactNode[] {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const key = `b${i}`;
    if (FENCE.test(line)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
      i++;
      blocks.push(
        <pre key={key}>
          <code>{body.join("\n")}</code>
        </pre>
      );
      continue;
    }
    const heading = line.match(HEADING);
    if (heading) {
      const Tag = `h${heading[1].length}` as "h1" | "h2" | "h3" | "h4" | "h5" | "h6";
      blocks.push(<Tag key={key}>{renderInline(heading[2], renderCitation, key)}</Tag>);
      i++;
      continue;
    }
    const numbered = line.match(NUMBERED);
    const list = BULLET.test(line) ? BULLET : numbered ? NUMBERED : null;
    if (list) {
      const items: ReactNode[] = [];
      let m: RegExpMatchArray | null;
      while (i < lines.length && (m = lines[i].match(list))) {
        const itemKey = `${key}.${items.length}`;
        items.push(<li key={itemKey}>{renderInline(m[m.length - 1], renderCitation, itemKey)}</li>);
        i++;
      }
      blocks.push(
        list === NUMBERED && numbered ? (
          <ol key={key} start={parseInt(numbered[1], 10)}>
            {items}
          </ol>
        ) : (
          <ul key={key}>{items}</ul>
        )
      );
      continue;
    }
    if (line.trim() === "") {
      i++;
      continue;
    }
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !FENCE.test(lines[i]) &&
      !HEADING.test(lines[i]) &&
      !BULLET.test(lines[i]) &&
      !NUMBERED.test(lines[i])
    ) {
      para.push(lines[i++]);
    }
    blocks.push(<p key={key}>{renderLines(para, renderCitation, key)}</p>);
  }
  return blocks;
}
