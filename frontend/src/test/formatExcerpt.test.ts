import { describe, expect, it } from "vitest";
import { formatExcerpt } from "../components/formatExcerpt";

describe("formatExcerpt", () => {
  it("joins lines broken mid-sentence", () => {
    expect(formatExcerpt("The memory state\ntensor is rich")).toBe("The memory state tensor is rich");
  });

  it("keeps bullets on their own lines", () => {
    expect(formatExcerpt("• Forget Gate\n• In a cell of\nthe network")).toBe("• Forget Gate\n• In a cell of the network");
  });

  it("rejoins a word split by a typographic hyphen", () => {
    expect(formatExcerpt("uninterpreta‐\nble features")).toBe("uninterpretable features");
  });

  it("leaves an ordinary hyphenated term alone", () => {
    expect(formatExcerpt("a well-known method")).toBe("a well-known method");
  });
});
