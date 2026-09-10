import { pretty } from '../format'

/**
 * Pretty-prints a JSON value as text. React escapes the content, so untrusted
 * trace text is never interpreted as markup.
 */
export function JsonBlock({ value, maxHeight }: { value: unknown; maxHeight?: number }) {
  if (value === null || value === undefined) return <p className="muted">none</p>
  return (
    <pre className="code-block" style={maxHeight ? { maxHeight } : undefined}>
      {pretty(value)}
    </pre>
  )
}

/** Plain text preserved with its whitespace. */
export function TextBlock({ text, className }: { text: string | null | undefined; className?: string }) {
  if (!text) return <p className="muted">empty</p>
  return <div className={`text-block${className ? ` ${className}` : ''}`}>{text}</div>
}
