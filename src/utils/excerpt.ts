/**
 * First prose paragraph of a markdown body, as plain text.
 * Skips headings, images, code/math blocks, quotes, lists, tables and HTML.
 */
export function firstParagraph(markdown: string): string {
  const lines: string[] = [];
  let fence = false;
  for (const raw of markdown.split('\n')) {
    const line = raw.trim();
    if (/^(```|~~~)/.test(line) || line === '$$') {
      fence = !fence;
      continue;
    }
    if (fence) continue;
    if (!line) {
      if (lines.length) break;
      continue;
    }
    if (/^(#|!\[|>|[-*+] |\d+\. |\||<|\$\$)/.test(line)) {
      if (lines.length) break;
      continue;
    }
    lines.push(line);
  }
  return (
    lines
      .join(' ')
      .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      // ponytail: only `**`, `__` and backticks; single `*`/`_` stay since they also appear in identifiers
      .replace(/\*\*|__|`/g, '')
      .trim()
  );
}
