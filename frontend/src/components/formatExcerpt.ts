/**
 * Reflows PDF text for reading. PDF extraction keeps the page's line breaks, which
 * cut sentences in a narrow panel. Bullets keep their own lines, and words split by
 * a typographic hyphen (U+2010) are rejoined. This affects display only.
 */
export function formatExcerpt(text: string): string {
  return text
    .replace(/‐\n/g, "")
    .replace(/\n(?!\s*[•▪◦*–-]\s)/g, " ")
    .replace(/ {2,}/g, " ");
}
