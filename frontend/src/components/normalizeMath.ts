const CODE = /(```[\s\S]*?```|`[^`\n]*`)/g;
const DISPLAY_ON_OWN_LINES = /^([ \t>]*)\\\[([\s\S]+?)\\\][ \t]*$/gm;
const REMAINING_DISPLAY = /\\\[([\s\S]+?)\\\]/g;
const INLINE = /\\\(([\s\S]+?)\\\)/g;

function oneLine(math: string): string {
  return math.trim().replace(/\s*\n\s*/g, " ");
}

/**
 * The model writes math as \( \) and \[ \], while remark-math reads $$. Display math
 * keeps its line's indentation, because removing it would end a surrounding list.
 * Code is left untouched. Single dollars stay off elsewhere, so "$5 and $10" remains
 * plain text.
 */
export function normalizeMath(markdown: string): string {
  return markdown
    .split(CODE)
    .map((part, index) => {
      if (index % 2 === 1) return part; // split() puts captured code at odd indexes.
      return part
        .replace(
          DISPLAY_ON_OWN_LINES,
          (_, indent: string, math: string) => `${indent}$$\n${indent}${oneLine(math)}\n${indent}$$`,
        )
        .replace(REMAINING_DISPLAY, (_, math: string) => `$$${oneLine(math)}$$`)
        .replace(INLINE, (_, math: string) => `$$${oneLine(math)}$$`);
    })
    .join("");
}
