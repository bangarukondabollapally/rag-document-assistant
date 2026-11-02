import "katex/dist/katex.min.css";
import { memo, useDeferredValue } from "react";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { normalizeMath } from "./normalizeMath";
import styles from "./AnswerBlock.module.css";

// Model output is untrusted. react-markdown drops raw HTML by default, and images
// are removed so an answer can't trigger a request to an outside server.
const DISALLOWED = ["img"];
const REMARK_PLUGINS: Parameters<typeof Markdown>[0]["remarkPlugins"] = [
  remarkGfm,
  [remarkMath, { singleDollarTextMath: false }],
];
const REHYPE_PLUGINS: Parameters<typeof Markdown>[0]["rehypePlugins"] = [
  [rehypeKatex, { throwOnError: false, strict: "ignore", trust: false }],
];

const Rendered = memo(function Rendered({ text }: { text: string }) {
  return (
    <Markdown
      remarkPlugins={REMARK_PLUGINS}
      rehypePlugins={REHYPE_PLUGINS}
      disallowedElements={DISALLOWED}
      components={{
        a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
        table: ({ node: _node, ...props }) => (
          <div className={styles.tableWrap}>
            <table {...props} />
          </div>
        ),
      }}
    >
      {normalizeMath(text)}
    </Markdown>
  );
});

export default function AnswerBlock({ text, streaming }: { text: string; streaming: boolean }) {
  // Keeps typing and scrolling responsive while a long answer re-renders per token.
  const deferred = useDeferredValue(text);
  return (
    <div className={styles.answer} aria-busy={streaming}>
      <Rendered text={deferred} />
    </div>
  );
}
