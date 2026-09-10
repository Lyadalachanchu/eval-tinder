function classify(line: string): string {
  if (line.startsWith('+++') || line.startsWith('---')) return 'diff-file'
  if (line.startsWith('@@')) return 'diff-hunk'
  if (line.startsWith('+')) return 'diff-add'
  if (line.startsWith('-')) return 'diff-del'
  return 'diff-ctx'
}

/** Renders a unified diff with +/- coloring. Each line is text content, never HTML. */
export function DiffView({ diff, emptyLabel = 'No differences.' }: { diff: string | null | undefined; emptyLabel?: string }) {
  if (!diff || !diff.trim()) return <p className="muted">{emptyLabel}</p>
  const lines = diff.replace(/\n$/, '').split('\n')
  return (
    <pre className="diff" aria-label="unified diff">
      {lines.map((line, i) => (
        <div key={i} className={`diff-line ${classify(line)}`}>
          {line === '' ? ' ' : line}
        </div>
      ))}
    </pre>
  )
}
