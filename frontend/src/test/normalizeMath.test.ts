import { describe, expect, it } from "vitest";
import { normalizeMath } from "../components/normalizeMath";

describe("normalizeMath", () => {
  it("converts inline and display delimiters", () => {
    expect(normalizeMath(String.raw`Let \(x_t\) be`)).toBe("Let $$x_t$$ be");
    expect(normalizeMath(String.raw`\[ a = b \]`)).toBe("$$\na = b\n$$");
  });

  it("keeps display math's indentation inside a list item", () => {
    const input = "1. Step\n   \\[\n   a = b\n   \\]\n   where it holds";
    expect(normalizeMath(input)).toBe("1. Step\n   $$\n   a = b\n   $$\n   where it holds");
  });

  it("treats display math sharing a line with text as inline", () => {
    expect(normalizeMath(String.raw`so \[a\], which holds`)).toBe("so $$a$$, which holds");
  });

  it("leaves code spans and fences alone", () => {
    const code = "`\\(x\\)` and\n```\n\\[y\\]\n```";
    expect(normalizeMath(code)).toBe(code);
  });

  it("leaves dollar amounts alone", () => {
    expect(normalizeMath("It cost $5 and $10.")).toBe("It cost $5 and $10.");
  });

  it("leaves an unclosed delimiter from a partial stream as it is", () => {
    expect(normalizeMath(String.raw`Let \(x_t`)).toBe(String.raw`Let \(x_t`);
  });
});
