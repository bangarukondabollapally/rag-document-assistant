import { describe, expect, it } from "vitest";
import { readNdjson } from "../api/client";

function streamOf(...parts: string[]) {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      parts.forEach((part) => controller.enqueue(encoder.encode(part)));
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>) {
  const items: unknown[] = [];
  for await (const item of readNdjson(stream)) items.push(item);
  return items;
}

describe("readNdjson", () => {
  it("reassembles lines split across network chunks", async () => {
    expect(await collect(streamOf('{"a":', '1}\n{"b"', ":2}\n"))).toEqual([{ a: 1 }, { b: 2 }]);
  });

  it("reads a final line with no trailing newline", async () => {
    expect(await collect(streamOf('{"a":1}'))).toEqual([{ a: 1 }]);
  });

  it("keeps multi-byte characters split across chunks intact", async () => {
    const bytes = new TextEncoder().encode('{"t":"λ"}\n');
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, 7));
        controller.enqueue(bytes.slice(7));
        controller.close();
      },
    });
    expect(await collect(stream)).toEqual([{ t: "λ" }]);
  });
});
